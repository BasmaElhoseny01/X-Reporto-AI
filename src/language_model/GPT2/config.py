
class Config:
    """
    Configuration class for a transformer model.

    Parameters:
    - max_seq_len (int): Maximum sequence length for input data.
    - d_model (int): Dimensionality of the model's hidden states.
    - d_ff1 (int): Dimensionality of the first feedforward layer.
    - d_ff2 (int): Dimensionality of the second feedforward layer.
    - d_ff3 (int): Dimensionality of the third feedforward layer.
    - dropout (float): Dropout rate to prevent overfitting.
    - num_heads (int): Number of attention heads in the model.
    - num_layers (int): Number of transformer layers in the model.
    - vocab_size (int): Size of the vocabulary used in the model.
    - ignore_index (int): Token index to be ignored during training.
    - bos_token (str): Beginning-of-sequence token.
    - eos_token (str): End-of-sequence token.
    - pad_token (str): Padding token.
    - pad_token_id (int): Token ID for padding.
    - bos_token_id (int): Token ID for the beginning-of-sequence.
    - eos_token_id (int): Token ID for the end-of-sequence.
    - pretrained_model (str): Pretrained model name or path.
    - use_checkpointing (bool): Whether to use gradient checkpointing for memory efficiency.
    - debug (bool): Enable debugging mode if True.
    """
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
