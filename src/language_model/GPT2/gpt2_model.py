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
import torch.utils.checkpoint


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
        layer_past: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None
        ):
        # print all inputs
        if self.config.debug:
            print("input_ids:", input_ids)
            print("layer_past:", layer_past)
            print("attention_mask:", attention_mask)
            print("position_ids:", position_ids)
            print("inputs_embeds:", inputs_embeds)
            print("image_hidden_states:", image_hidden_states)
            print("labels:", labels)
            print("use_cache:", use_cache)
            print("output_attentions:", output_attentions)
            # print shape of all inputs
            if input_ids is not None:
                print("input_ids shape:", input_ids.shape)
            if layer_past is not None:
                print("layer_past shape:", layer_past[0][0].shape)
            if attention_mask is not None:
                print("attention_mask shape:", attention_mask.shape)
            if position_ids is not None:
                print("position_ids shape:", position_ids.shape)
            if inputs_embeds is not None:
                print("inputs_embeds shape:", inputs_embeds.shape)
            if image_hidden_states is not None:
                print("image_hidden_states shape:", image_hidden_states.shape)
            if labels is not None:
                print("labels shape:", labels.shape)
            
        if image_hidden_states is not None:
            # convert image hidden states dtype to dtype of the model
            image_hidden_states = image_hidden_states.to(dtype=self.fc.weight.dtype)
            # print("image_hidden_states dtype:", image_hidden_states.dtype)
            image_hidden_states = self.image_to_text(image_hidden_states)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        
        # apply embedding layers
        if inputs_embeds is None:
            hidden_states = self.wte(input_ids) # (batch_size, seq_len, d_model)

        if position_ids is None:
            # apply positional encoding layer
            # convert hidden states dtype to dtype of the model
            hidden_states = hidden_states.to(dtype=self.fc.weight.dtype) 
            hidden_states = self.positional_encoding(hidden_states) # (batch_size, seq_len, d_model)

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

        presents = ()
        if use_cache is False:
            # make presents be None
            for i in range(self.num_layers):
                presents = presents + (None,)

        if layer_past is None:
            # make layer_past be None
            layer_past = ()
            for i in range(self.num_layers):
                layer_past = layer_past + (None,)
        # apply transformer blocks
        for i in range(self.num_layers):
            # check if gradient checkpointing should be used
            if self.config.use_checkpointing:
                if self.config.debug and i == 0:
                    print("using gradient checkpointing")
                outputs = torch.utils.checkpoint.checkpoint(self.blocks[i], hidden_states,attention_mask, image_hidden_states,layer_past[i],use_cache,output_attentions)
                hidden_states = outputs[0]
            else:
                outputs = self.blocks[i](hidden_states, image_hidden_states=image_hidden_states,attention_mask=attention_mask,layer_past=layer_past[i],use_cache=use_cache,output_attentions=output_attentions)
                hidden_states = outputs[0]
            if use_cache:
                present = outputs[1]
                presents = presents + (present,)

            if self.config.debug and i == self.num_layers - 1:
                print("memory usage after block", i, ":", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
            
        # compute model output logits
        logits = self.fc(hidden_states)

        if self.config.debug:
            # wait two seconds
            wait(4)
            # print memory usage
            print("memory usage after logits:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
            print("logits dtype:", logits.dtype)
            print("present shape:", presents[0][0].shape)
            # print("presents", presents)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # labels = labels.to(logits.device)
            # convert labels dtype to dtype of the model
            # labels = labels.to(dtype=torch.long)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            del logits
            shift_labels = labels[..., 1:].contiguous()
            del labels

            if self.config.debug:
                # wait two seconds
                wait(2)
                # print memory usage
                print("memory usage before loss:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
            
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            # convert logits dtype to float32
            shift_logits = shift_logits.to(dtype=torch.float32)
            if self.config.debug:
                # wait two seconds
                wait(2)
                
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss,shift_logits) 
        if use_cache:
            return (logits, presents)
        return (logits,)
    # add no_grad to forward pass
    @torch.no_grad()
    def generate(self, max_length, image_hidden_states,device):
        
        batch_size = image_hidden_states.size(0)

        input_ids = torch.full(size=(batch_size, 1), fill_value=self.config.bos_token_id, dtype=torch.int64, device=device)
        model_kwargs = {"attention_mask": torch.ones(size=(batch_size, 1), dtype=torch.int64, device=device),
                        "use_cache": True}
        
        #greedy search
        seq_len = 1
        #start with a random token
        all_sequences_to_generate = torch.ones(size=(batch_size,), dtype=torch.int64, device=device)
        cur_len = seq_len
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next
            logits, presents = self.forward(**model_inputs, image_hidden_states=image_hidden_states)
            next_token_logits = logits[:, -1, :]  # of shape [batch_size x vocab_size]
            # greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)
            # concatenate the new token
            next_token = next_token * all_sequences_to_generate + self.config.pad_token_id * (1 - all_sequences_to_generate)
            
            # update input_ids, attention mask and length for next step
            input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
            # update model kwargs
            print("**model_kwargs",model_kwargs)
            model_kwargs = self.update_model_kwargs(presents,model_kwargs)
            # update sequence length
            cur_len += 1

            # if eos_token was found in one sentence, set sentence to finished
            binary_mask = (next_token != self.config.eos_token_id).long()
            all_sequences_to_generate = all_sequences_to_generate.mul(binary_mask)

            # stop when all sentences are finished or if we exceed the maximum length
            if all_sequences_to_generate.max() == 0 or (max_length and cur_len >= max_length):
                break
        return input_ids



    def prepare_inputs_for_generation(self, input_ids, layer_past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if layer_past:
            past_length = layer_past[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if layer_past:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "layer_past": layer_past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
        }
    def update_model_kwargs(self, presents, model_kwargs):
        model_kwargs["past"] = presents
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        return model_kwargs
    
    # def update_model_kwargs(self, input_ids, layer_past=None, inputs_embeds=None, **kwargs):
    #     token_type_ids = kwargs.get("token_type_ids", None)
    #     # Omit tokens covered by past_key_values
    #     if layer_past:
    #         past_length = layer_past[0][0].shape[2]

    #         # Some generation methods already pass only the last input ID
    #         if input_ids.shape[1] > past_length:
    #             remove_prefix_length = past_length
    #         else:
    #             # Default to old behavior: keep only final ID
    #             remove_prefix_length = input_ids.shape[1] - 1

    #         input_ids = input_ids[:, remove_prefix_length:]
    #         if token_type_ids is not None:
    #             token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

    #     attention_mask = kwargs.get("attention_mask", None)
    #     position_ids = kwargs.get("position_ids", None)

    #     if attention_mask is not None and position_ids is None:
    #         # create position_ids on the fly for batch generation
    #         position_ids = attention_mask.long().cumsum(-1) - 1
    #         position_ids.masked_fill_(attention_mask == 0, 1)
    #         if layer_past:
    #             position_ids = position_ids[:, -input_ids.shape[1] :]
    #     else:
    #         position_ids = None

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and layer_past is None:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids}

    #     model_inputs.update(
    #         {
    #             "layer_past": layer_past,
    #             "use_cache": kwargs.get("use_cache"),
    #             "position_ids": position_ids,
    #             "attention_mask": attention_mask,
    #             # "token_type_ids": token_type_ids,
    #         }
    #     )

    #     return model_inputs


def wait(seconds):
    seconds = seconds * 200000000
    for i in range(seconds):
        continue
    
    
def test(use_checkpointing = True, debug=True,batch_size = 4,seq_length =1024,is_half = False):
    config = Config()
    config.d_model = 768
    config.d_ff = 768
    config.num_layers = 12
    config.vocab_size = 50257
    config.max_seq_len = seq_length
    config.pretrained_model = "gpt2"
    config.use_checkpointing = use_checkpointing
    config.debug = debug
    image_config = Config()
    image_config.d_model = 768
    image_config.d_ff1 = 768
    image_config.d_ff2 = 768
    image_config.d_ff3 = 768
    
    model = CustomGPT2(config,image_config)
    if is_half:
        model.half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = torch.randint(0, 50257, (batch_size, seq_length))
    image_hidden_states = torch.randn(batch_size, 1, config.d_model)
    labels = torch.randint(0, 50257, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    x = x.to(device)
    image_hidden_states = image_hidden_states.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    model.train()
    for i in range(4):
        loss, logits = model(x,layer_past=None, image_hidden_states = image_hidden_states,attention_mask = attention_mask, labels=labels, use_cache = True,output_attentions = False)
        logits = logits.to("cpu")
        del logits
        print("loss:", loss.item())
        print("memory usage after loss:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
        # apply optimizer
        # wait two seconds
        wait(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer.step()
        print("memory usage after step:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
    # move to cpu
    model.to("cpu")
    x = x.to("cpu")
    image_hidden_states = image_hidden_states.to("cpu")
    attention_mask = attention_mask.to("cpu")
    labels = labels.to("cpu")
    # free memory
    del x
    del image_hidden_states
    del attention_mask
    del labels
    torch.cuda.empty_cache()
    del model
    # apply garbage collection
    import gc
    gc.collect()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    # print memory usage
    print("memory usage after empty cache:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")


def test_genertation(use_checkpointing = True, debug=True,batch_size = 4,seq_length =1024,is_half = False):
    config = Config()
    config.d_model = 768
    config.d_ff = 768
    config.num_layers = 12
    config.vocab_size = 50257
    config.max_seq_len = seq_length
    config.pretrained_model = "gpt2"
    config.use_checkpointing = use_checkpointing
    config.debug = debug
    image_config = Config()
    image_config.d_model = 768
    image_config.d_ff1 = 768
    image_config.d_ff2 = 768
    image_config.d_ff3 = 768
    
    model = CustomGPT2(config,image_config)
    if is_half:
        model.half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_hidden_states = torch.randn(batch_size, 1, config.d_model)
    image_hidden_states = image_hidden_states.to(device)
    model.eval()
    generated = model.generate(3, image_hidden_states,device)
    print("generated: ",generated)
    
if __name__ == '__main__':

    test_genertation(use_checkpointing = True, debug=True,batch_size = 1,seq_length =1024,is_half = False)

    # print("Test 1")
    # print("use_checkpointing = True, debug=True,batch_size = 4,seq_length =1024,is_half = False")
    # test(use_checkpointing = True, debug=True,batch_size = 4,seq_length =1024,is_half = False)
    # # wait two seconds
    # wait(2)
    # # print memory usage
    # print("memory usage after test 1:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
    # print("Test 2")
    # print("use_checkpointing = True, debug=True,batch_size = 4,seq_length =1024,is_half = True")
    # test(use_checkpointing = True, debug=True,batch_size = 6,seq_length =1024,is_half = True)