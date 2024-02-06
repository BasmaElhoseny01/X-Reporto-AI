from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
import math
from src.language_model.GPT2.feed_forward import FeedForward
from src.language_model.GPT2.layer_normalization import LayerNormalization
from src.language_model.GPT2.residual_connection import ResidualConnection
from src.language_model.GPT2.gpt2_attention import CustomGPT2MultiHeadAttention
from src.language_model.GPT2.positional_encoding import PositionalEncoding
from src.language_model.GPT2.embeddings import InputEmbedding
from src.language_model.GPT2.gpt2_block import CustomGPT2Block
from src.language_model.GPT2.config import Config

class CustomGPT2(nn.Module):
    def __init__(self, config,image_config):
        super(CustomGPT2, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.ignore_index = config.ignore_index
        self.pretrained_model = config.pretrained_model
        # define image transformation feed forward layer
        self.image_to_text = FeedForward(image_config)

        # define embedding layers
        self.wte = InputEmbedding(self.config)
        # self.wte = nn.Embedding(config.vocab_size, self.d_model)
        # self.wpe = nn.Embedding(self.d_model, self.d_model)
        self.drop = nn.Dropout(self.config.dropout)

        # define positional encoding layer
        self.positional_encoding = PositionalEncoding(self.config)

        # define transformer blocks
        self.blocks = nn.ModuleList([CustomGPT2Block(self.config) for _ in range(self.config.num_layers)])

        # define layer normalization layer
        self.ln = LayerNormalization(self.config)

        # define fully connected layer
        self.fc = nn.Linear(self.d_model, self.vocab_size)
        self.init_weights()
        self.load_pretrained_weights()
        
    def init_weights(self):
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()
    def convert_to_half(self):
        self.fc.weight.data = self.fc.weight.data.half()
        self.fc.bias.data = self.fc.bias.data.half()
        for i in range(self.num_layers):
            self.blocks[i].attn.c_attn.weight.data = self.blocks[i].attn.c_attn.weight.data.half()
            self.blocks[i].attn.c_attn.bias.data = self.blocks[i].attn.c_attn.bias.data.half()
            self.blocks[i].attn.c_proj.weight.data = self.blocks[i].attn.c_proj.weight.data.half()
            self.blocks[i].attn.c_proj.bias.data = self.blocks[i].attn.c_proj.bias.data.half()
            self.blocks[i].rc1.ln.gamma.data = self.blocks[i].rc1.ln.gamma.data.half()
            self.blocks[i].rc1.ln.beta.data = self.blocks[i].rc1.ln.beta.data.half()
            self.blocks[i].rc2.ln.gamma.data = self.blocks[i].rc2.ln.gamma.data.half()
            self.blocks[i].rc2.ln.beta.data = self.blocks[i].rc2.ln.beta.data.half()
            self.blocks[i].ff.fc1.weight.data = self.blocks[i].ff.fc1.weight.data.half()
            self.blocks[i].ff.fc1.bias.data = self.blocks[i].ff.fc1.bias.data.half()
            self.blocks[i].ff.fc2.weight.data = self.blocks[i].ff.fc2.weight.data.half()
            self.blocks[i].ff.fc2.bias.data = self.blocks[i].ff.fc2.bias.data.half()

    def load_pretrained_weights(self):
        if self.pretrained_model is not None:
            # use GPT2 model with language modeling head, since we want to generate phrases
            gpt_with_lm_head = GPT2LMHeadModel.from_pretrained(self.pretrained_model)

            # copy weights from pre-trained model to custom model
            # print("pretrained model architecture: ", gpt_with_lm_head)

            # copy weights of embedding layers
            self.wte.token_embedding.weight.data = gpt_with_lm_head.transformer.wte.weight.data
            
            # copy weights of transformer blocks
            for i in range(self.num_layers):
                self.blocks[i].attn.c_attn.weight.data = gpt_with_lm_head.transformer.h[i].attn.c_attn.weight.data
                self.blocks[i].attn.c_attn.bias.data = gpt_with_lm_head.transformer.h[i].attn.c_attn.bias.data
                self.blocks[i].attn.c_proj.weight.data = gpt_with_lm_head.transformer.h[i].attn.c_proj.weight.data
                self.blocks[i].attn.c_proj.bias.data = gpt_with_lm_head.transformer.h[i].attn.c_proj.bias.data
                self.blocks[i].rc1.ln.gamma.data = gpt_with_lm_head.transformer.h[i].ln_1.weight.data
                self.blocks[i].rc1.ln.beta.data = gpt_with_lm_head.transformer.h[i].ln_1.bias.data
                self.blocks[i].rc2.ln.gamma.data = gpt_with_lm_head.transformer.h[i].ln_2.weight.data
                self.blocks[i].rc2.ln.beta.data = gpt_with_lm_head.transformer.h[i].ln_2.bias.data
                self.blocks[i].ff.fc1.weight.data = gpt_with_lm_head.transformer.h[i].mlp.c_fc.weight.data.T
                self.blocks[i].ff.fc1.bias.data = gpt_with_lm_head.transformer.h[i].mlp.c_fc.bias.data
                self.blocks[i].ff.fc2.weight.data = gpt_with_lm_head.transformer.h[i].mlp.c_proj.weight.data.T
                self.blocks[i].ff.fc2.bias.data = gpt_with_lm_head.transformer.h[i].mlp.c_proj.bias.data

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None
        ):

        if image_hidden_states is not None:
            # convert image hidden states dtype to dtype of the model
            image_hidden_states = image_hidden_states.to(dtype=self.fc.weight.dtype)
            print("image_hidden_states dtype:", image_hidden_states.dtype)
            image_hidden_states = self.image_to_text(image_hidden_states)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        
        # apply embedding layers
        if inputs_embeds is None:
            hidden_states = self.wte(input_ids)

        if position_ids is None:
            # apply positional encoding layer
            # convert hidden states dtype to dtype of the model
            hidden_states = hidden_states.to(dtype=self.fc.weight.dtype)
            print("hidden_states dtype:", hidden_states.dtype)
            hidden_states = self.positional_encoding(hidden_states)

        # apply dropout layer
        hidden_states = self.drop(hidden_states)

        # create attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            attention_mask = attention_mask[:, None, None, :]
            # convert attention mask of shape (batch_size,1,1, max_seq_len) to (batch_size, 1, 1, 1+max_seq_len) by concatenating 1s
            ones = torch.ones(attention_mask.size()[:-1] + (1,), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((ones, attention_mask), dim=-1)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # apply transformer blocks
        for i in range(self.num_layers):
            hidden_states = self.blocks[i](hidden_states, image_hidden_states=image_hidden_states,attention_mask=attention_mask)
        
        # compute model output logits
        logits = self.fc(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # convert labels dtype to dtype of the model
            labels = labels.to(dtype=torch.long)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            # convert logits dtype to float32
            # shift_logits = shift_logits.to(dtype=torch.float32)

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss,logits) 
        return logits

if __name__ == '__main__':
    # Test
    config = Config()
    config.d_model = 768
    config.d_ff = 768
    config.num_layers = 12
    config.vocab_size = 50257
    config.max_seq_len = 1024
    config.pretrained_model = "gpt2"

    image_config = Config()
    image_config.d_model = 768
    image_config.d_ff1 = 768
    image_config.d_ff2 = 768
    image_config.d_ff3 = 768
    image_config.num_heads = 8
    image_config.num_layers = 6
    image_config.vocab_size = 50257
    image_config.max_seq_len = 1024
    image_config.dropout = 0.1
    gpt2 = CustomGPT2(config,image_config)
    x = torch.randint(0, 50257, (2, 5))
    image_hidden_states = torch.randn(2, 1, config.d_model)
    # create attention mask
    attention_mask = torch.ones(2, 5)
    print(gpt2(x, image_hidden_states = image_hidden_states,attention_mask = attention_mask).size()) # torch.Size([2, 5, 50257])
    print(gpt2)
    x = torch.randint(0, 50257, (2, 5))
    image_hidden_states = torch.randn(2, 1, config.d_model)
    labels = torch.randint(0, 50257, (2, 5))
    print(gpt2(x, image_hidden_states = image_hidden_states,attention_mask = attention_mask, labels=labels)[0]) # tensor(13.5676, grad_fn=<NllLossBackward>)
    print(gpt2)

    # save model
    torch.save(gpt2.state_dict(), "gpt2.pth")

    # convert model to half precision
    gpt2.half()

    # print model dtype
    print(gpt2.fc.weight.dtype)
    x = torch.randint(0, 50257, (2, 5))
    image_hidden_states = torch.randn(2, 1, config.d_model)
    labels = torch.randint(0, 50257, (2, 5))
    print(gpt2(x, image_hidden_states = image_hidden_states,attention_mask = attention_mask, labels=labels)[0]) # tensor(13.5676, grad_fn=<NllLossBackward>)
    print(gpt2)

    # save half precision model
    torch.save(gpt2.state_dict(), "gpt2_half.pth")

    # convert model to half precision
    gpt2.half()

    # print model dtype
    print(gpt2.fc.weight.dtype)
    # save half precision model
    torch.save(gpt2.state_dict(), "gpt2_half2.pth")

    