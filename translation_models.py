import os
import logging
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from IndicTransToolkit import IndicProcessor  # Add this import

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TranslationModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = {
            'mbart50': 'facebook/mbart-large-50-many-to-many-mmt',
            'indictrans': 'ai4bharat/indictrans2-en-indic-1b'
        }
        self.current_model = None
        self.ip = None  # Add IndicProcessor instance

    def load_model(self, model_choice='mbart50'):
        if model_choice not in self.model_name:
            logger.warning(f"Invalid model choice {model_choice}, defaulting to mbart50")
            model_choice = 'mbart50'

        if self.current_model == model_choice and self.model is not None:
            logger.info(f"Model {model_choice} already loaded")
            return

        logger.info(f"Loading {model_choice} model and tokenizer")
        if model_choice == 'mbart50':
            self.model = MBartForConditionalGeneration.from_pretrained(self.model_name['mbart50']).to(self.device)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name['mbart50'])
        elif model_choice == 'indictrans':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name['indictrans'],
                trust_remote_code=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name['indictrans'],
                trust_remote_code=True
            )
            self.ip = IndicProcessor(inference=True)  # Initialize IndicProcessor
        
        self.current_model = model_choice
        logger.info(f"{model_choice} model loaded successfully")

    def translate(self, text, source_lang='en', target_lang='hi', model_choice='mbart50'):
        if not text.strip():
            return "No input text to translate"

        self.load_model(model_choice)
        
        logger.debug(f"Translating with {model_choice}: '{text}'")
        if model_choice == 'mbart50':
            self.tokenizer.src_lang = "en_XX"
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            translated_tokens = self.model.generate(
                **inputs, 
                forced_bos_token_id=self.tokenizer.lang_code_to_id["hi_IN"],
                max_length=512
            )
        elif model_choice == 'indictrans':
            # Use IndicProcessor for preprocessing
            batch = self.ip.preprocess_batch(
                [text],  # Single sentence as a list
                src_lang="eng_Latn",
                tgt_lang="hin_Deva"
            )
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                    use_cache=True,
                    min_length=0
                )
            with self.tokenizer.as_target_tokenizer():
                translated_text = self.tokenizer.batch_decode(
                    translated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]  # Take first item since batch size is 1
            translated_text = self.ip.postprocess_batch([translated_text], lang="hin_Deva")[0]  # Postprocess

        if model_choice == 'mbart50':
            translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        logger.debug(f"Raw translated tokens: {translated_tokens}, Decoded: {translated_text}")

        # Clear PyTorch cache to free up GPU/CPU memory
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        logger.debug("Cleared PyTorch cache after translation")

        # Explicitly delete temporary objects to free memory
        del inputs, translated_tokens
        import gc
        gc.collect()
        logger.debug("Garbage collection performed after translation")

        return translated_text