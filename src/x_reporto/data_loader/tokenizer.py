from transformers import GPT2Tokenizer

class Tokenizer:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self,phrases):
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]
        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return self.tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    
