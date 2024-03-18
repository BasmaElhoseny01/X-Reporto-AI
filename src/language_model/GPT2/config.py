
class Config:
    max_seq_len = 1024
    d_model = 1024
    d_ff1 = 1024
    d_ff2 = 1024
    d_ff3 = 1024
    dropout = 0.1
    num_heads = 16
    num_layers = 24
    vocab_size = 50257
    ignore_index = -100
    bos_token = "<|endoftext|>"  
    eos_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"
    pad_token_id = 50256
    bos_token_id = 50256
    eos_token_id = 50256
    pretrained_model = "healx/gpt-2-pubmed-medium"
    use_checkpointing = True
    debug = False
