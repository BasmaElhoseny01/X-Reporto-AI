from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.language_model.GPT2.feed_forward import FeedForward
from src.language_model.GPT2.residual_connection import ResidualConnection
from src.language_model.GPT2.gpt2_attention import CustomGPT2MultiHeadAttention

class CustomGPT2Block(nn.Module):
    """
    Custom block for the GPT-2 model, consisting of a multi-head attention layer,
    feedforward layer, and residual connections.

    Args:
        config (Config): An instance of the configuration class with model hyperparameters.
    """

    def __init__(self, config):
        """
        Initializes the CustomGPT2Block.

        Parameters:
        - config (Config): An instance of the configuration class with model hyperparameters.
        """
        super(CustomGPT2Block, self).__init__()
        self.config = config
        self.rc1 = ResidualConnection(self.config)
        self.attn = CustomGPT2MultiHeadAttention(self.config)
        self.rc2 = ResidualConnection(self.config)
        self.ff = FeedForward(self.config)

    def forward(self, x:Optional[torch.Tensor]=None, attention_mask:Optional[torch.Tensor]=None, image_hidden_states:Optional[torch.Tensor]=None, layer_past=Optional[Tuple[torch.Tensor]], use_cache:Optional[bool]=False, output_attentions:Optional[bool]=False):
        """
        Forward pass of the CustomGPT2Block.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - attention_mask (torch.Tensor): Attention mask for masking out padded tokens.
        - image_hidden_states (torch.Tensor, optional): Hidden states from image processing.
        - layer_past (tuple, optional): Cached past attention weights.
        - use_cache (bool, optional): Whether to use caching for attention weights.
        - output_attentions (bool, optional): Whether to output attention weights.

        Returns:
        - Tuple: Output tuple containing the transformed tensor, cached present states (if use_cache),
          and attention weights (if output_attentions).
        """
        """
        Some Explaintion:
        rc1 and rc2: ResidualConnection modules that apply layer normalization and dropout to the output of their respective sublayers.

        attn: CustomGPT2MultiHeadAttention, which is a multi-head attention layer with layer normalization and dropout.

        ff: FeedForward layer, which consists of linear transformations and activation functions.

        The forward method applies these layers sequentially, incorporating residual connections. The attention layer (attn) is applied within the first residual connection (rc1), and the feedforward layer (ff) is applied within the second residual connection (rc2).

        The method returns a tuple containing the transformed tensor, present states (if use_cache is True), and attention weights (if output_attentions is True).
        """
        # Apply residual connection to attention layer
        outputs = self.rc1(x, lambda x: self.attn(x, image_hidden_states=image_hidden_states,
                                                  attention_mask=attention_mask,
                                                  layer_past=layer_past,
                                                  use_cache=use_cache,
                                                  output_attentions=output_attentions),
                           is_attention=True)
        x = outputs[0]  # Output of the attention layer
        outputs = outputs[1:]  # Present states, output_attentions

        # Apply residual connection to feedforward layer
        x = self.rc2(x, lambda x: self.ff(x))

        if output_attentions is False:
            outputs = (outputs[0],)  # Tuple (present,)
        if use_cache:
            outputs = (x,) + outputs[0:]
            return outputs  # Tuple (x, present, output_attentions)
        return (x,) + outputs[1:]  # Tuple (x, output_attentions)


if __name__ == '__main__':
    # Test
    from src.language_model.GPT2.config import Config
    config = Config()
    # config.d_model = 512
    # config.d_ff = 2048
    # config.num_heads = 8
    # config.dropout = 0.1
    gpt2_block = CustomGPT2Block(config)
    x = torch.randn(2, 5, config.d_model)
    attention_mask = torch.ones(2, 5, dtype=x.dtype, device=x.device)
    attention_mask = attention_mask[:, None, None, :]
    # attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * -10000.0

    image_hidden_states = torch.randn(2, 1, config.d_model)
    print(gpt2_block(x,attention_mask)[0].size()) # torch.Size([2, 5, 512])
    print(gpt2_block)
    x = torch.randn(2, 5, config.d_model)
    ones = torch.ones(attention_mask.size()[:-1] + (1,), dtype=attention_mask.dtype, device=attention_mask.device)
    attention_mask = torch.cat((ones, attention_mask), dim=-1)
    # attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * -10000.0
    image_hidden_states = torch.randn(2, 1, config.d_model)
    print(gpt2_block(x,attention_mask, image_hidden_states)[0].size()) # torch.Size([2, 5, 512])
    print(gpt2_block)
    