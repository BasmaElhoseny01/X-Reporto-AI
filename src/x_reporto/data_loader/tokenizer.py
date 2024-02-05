from transformers import GPT2Tokenizer
from src.language_model.GPT2.config import Config
class Tokenizer:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        self.tokenizer.pad_token = Config.pad_token

    def __call__(self,phrases):
        bos_token = Config.bos_token  
        eos_token = Config.eos_token

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]
        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return self.tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    
