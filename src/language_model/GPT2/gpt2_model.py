import gc
from typing import Any, Optional, Tuple
import torch
from torch import Tensor
from torch import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
from src.language_model.GPT2.feed_forward import FeedForward
from src.language_model.GPT2.layer_normalization import LayerNormalization
from src.language_model.GPT2.positional_encoding import PositionalEncoding
from src.language_model.GPT2.embeddings import InputEmbedding
from src.language_model.GPT2.gpt2_block import CustomGPT2Block
from src.language_model.GPT2.config import Config
import torch.utils.checkpoint
from transformers import GPT2Tokenizer
from transformers.generation.beam_search import BeamSearchScorer
from torchsummary import summary

import sys
class CustomGPT2(nn.Module):
    def __init__(self, config,image_config):
        """
        Custom GPT-2 model with additional modifications.

        Args:
            config (Config): An instance of the Config class containing model configuration.
            image_config (Any): Configuration for image transformation feed-forward layer.

        Attributes:
            config (Config): Model configuration.
            d_model (int): Dimension of the model.
            num_layers (int): Number of transformer blocks.
            vocab_size (int): Size of the vocabulary.
            ignore_index (int): Index to ignore during loss calculation.
            pretrained_model (Optional[str]): Path to a pre-trained GPT-2 model.
            image_to_text (FeedForward): Image transformation feed-forward layer.
            wte (InputEmbedding): Embedding layer for input tokens.
            drop (nn.Dropout): Dropout layer.
            positional_encoding (PositionalEncoding): Positional encoding layer.
            blocks (nn.ModuleList): List of transformer blocks.
            ln (LayerNormalization): Layer normalization layer.
            fc (nn.Linear): Fully connected layer for model output.
        """
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
        self.tokenizer = GPT2Tokenizer.from_pretrained("healx/gpt-2-pubmed-medium")
        
    def init_weights(self):
        """
        initialize model weights.
        """
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()
    def convert_to_half(self):
        """
        Convert model parameters to half precision (float16).
        """
        
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
        """
        Load weights from a pre-trained GPT-2 model.
        """
        if self.pretrained_model is not None:
            # use GPT2 model with language modeling head, since we want to generate phrases
            gpt_with_lm_head = GPT2LMHeadModel.from_pretrained(self.pretrained_model)

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
            
            print("loaded pre-trained weights successfully")

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        seq_len: Optional[int] = None
        ):
        """
        Forward pass of the CustomGPT2 model.

        Args:
            input_ids (Optional[torch.LongTensor]): Input token IDs.
            layer_past (Optional[Tuple[Tuple[torch.Tensor]]]): Past layers for autoregressive generation.
            attention_mask (Optional[torch.FloatTensor]): Attention mask for input tokens.
            position_ids (Optional[torch.LongTensor]): Positional IDs for input tokens.
            inputs_embeds (Optional[torch.FloatTensor]): Embedded input tokens.
            image_hidden_states (Optional[torch.Tensor]): Hidden states from image processing.
            labels (Optional[torch.LongTensor]): Target labels for training.
            use_cache (Optional[bool]): Whether to use past layers for autoregressive generation.
            output_attentions (Optional[bool]): Whether to output attention weights.
            seq_len (Optional[int]): Length of the input sequence.

        Returns:
            Tuple[torch.Tensor or Tuple[torch.Tensor]]: Model output, and optionally, past layers and attention weights.
        """
            
        if image_hidden_states is not None:
            # convert image hidden states dtype to dtype of the model
            image_hidden_states = image_hidden_states.to(dtype=self.fc.weight.dtype)
            image_hidden_states = self.image_to_text(image_hidden_states)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        # if torch.isnan(input_ids).any():
        #     print("error: input_ids contains nan")
        #     sys.exit()
        
        # apply embedding layers
        if inputs_embeds is None:
            hidden_states = self.wte(input_ids) # (batch_size, seq_len, d_model)

        # if torch.isnan(hidden_states).any():
        #     print("1 error: hidden_states contains nan")
        #     sys.exit()

        if position_ids is None:
            # apply positional encoding layer
            # convert hidden states dtype to dtype of the model
            hidden_states = hidden_states.to(dtype=self.fc.weight.dtype) 
            hidden_states = self.positional_encoding(hidden_states,seq_len) # (batch_size, seq_len, d_model)

        # create attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            attention_mask = attention_mask[:, None, None, :]
            # convert attention mask of shape (batch_size,1,1, max_seq_len) to (batch_size, 1, 1, 1+max_seq_len) by concatenating 1s
            ones = torch.ones(attention_mask.size()[:-1] + (1,), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((ones, attention_mask), dim=-1)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0  #do masking

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
                outputs = torch.utils.checkpoint.checkpoint(self.blocks[i], hidden_states,attention_mask, image_hidden_states,layer_past[i],use_cache,output_attentions)
                hidden_states = outputs[0]

            else:
                outputs = self.blocks[i](hidden_states, image_hidden_states=image_hidden_states,attention_mask=attention_mask,layer_past=layer_past[i],use_cache=use_cache,output_attentions=output_attentions)
                hidden_states = outputs[0]
            if use_cache:
                present = outputs[1]
                presents = presents + (present,)
            
        

        # compute model output logits
        logits = self.fc(hidden_states) 

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

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            # convert logits dtype to float32

            shift_logits = shift_logits.to(dtype=torch.float32)

                
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss,shift_logits) 
        if use_cache:
            return (logits, presents)
        return (logits,)

    def prepare_inputs_for_generation(self, input_ids:Tensor=None, seq_len:int=1, **kwargs:dict[str, Any]):
        """
        Prepares inputs for the generation process.

        Args:
            input_ids (Tensor): Input token IDs.
            seq_len (int): Length of the sequence.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            dict: Model inputs including input IDs, past layers, use_cache flag, position IDs, attention mask, and sequence length.
        """
        # Omit tokens covered by past_key_values
        layer_past = kwargs.get("layer_past", None)
        if layer_past:
            past_length = layer_past[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            # if token_type_ids is not None:
            #     token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        position_ids = None
        return {
            "input_ids": input_ids,
            "layer_past": layer_past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "seq_len": seq_len
        }

    def update_model_kwargs(self, model_kwargs, presents:Any=None):
        """
        Updates model keyword arguments during generation.

        Args:
            model_kwargs (dict): Model keyword arguments.
            presents (Any): Past layers.

        Returns:
            dict: Updated model keyword arguments.
        """
        model_kwargs["layer_past"] = presents
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        return model_kwargs
    
    def generate(self, max_length: int = 300, image_hidden_states: Tensor = None, Temperature: int = 1,
             top_k: int = 0, device: device = None, greedy: bool = False, sampling: bool = False,
             sampling_top_k: bool = False) -> Tensor:
        """
        Generates sequences using greedy decoding, sampling, or top-k sampling.

        Args:
            max_length (int): Maximum length of the generated sequence.
            image_hidden_states (Tensor): Hidden states from image processing.
            Temperature (int): Controls the randomness of sampling (higher values make output more random).
            top_k (int): Number of top tokens to consider during top-k sampling.
            device (device): Device to perform generation on.
            greedy (bool): Whether to use greedy decoding.
            sampling (bool): Whether to use random sampling.
            sampling_top_k (bool): Whether to use top-k sampling.

        Returns:
            Tensor: Generated sequence of token IDs.
        """
        batch_size = image_hidden_states.size(0)

        input_ids = torch.full(size=(batch_size, 1), fill_value=self.config.bos_token_id, dtype=torch.int64, device=device)
        model_kwargs = {"attention_mask": torch.ones(size=(batch_size, input_ids.shape[-1]), dtype=torch.int64, device=device),
                        "use_cache": True}

        # greedy search
        seq_len = input_ids.shape[-1]
        # start with a random token
        all_sequences_to_generate = torch.ones(size=(batch_size,), dtype=torch.int64, device=device)  # (batch_size,)
        cur_len = seq_len
        while True:
            # prepare model inputs (attention mask, layer_past, inputs_ids, position_ids, use_cache)
            model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids, seq_len=cur_len, **model_kwargs)
            # forward pass to get next
            logits, presents = self.forward(**model_inputs, image_hidden_states=image_hidden_states)
            next_token_logits = logits[:, -1, :]  # of shape [batch_size x vocab_size]

            # greedy decoding
            if greedy:
                next_token = torch.argmax(next_token_logits, dim=-1)  # of shape [batch_size]
            # sampling
            elif sampling:
                next_token = torch.multinomial(F.softmax(next_token_logits / Temperature, dim=-1), num_samples=1).squeeze(1)
            # top-k sampling
            elif sampling_top_k:  
                top_k_values, top_k_indices = torch.topk(next_token_logits /Temperature, top_k, dim=-1)
                # Apply softmax to the top-k values
                top_k_probs = F.softmax(top_k_values, dim=-1)
                # Multinomial sampling based on the top-k probabilities
                sampled_index = torch.multinomial(top_k_probs, 1).item()
                next_token=top_k_indices[0, sampled_index].item()
                # print top k tokens
                # print("top_k_indices: ",top_k_indices)
                tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices[0])
                # print("top_k_tokens: ",tokens)
                # print("top_k_indices: ",top_k_indices)
            # concatenate the new token
            next_token = next_token * all_sequences_to_generate + self.config.pad_token_id * (1 - all_sequences_to_generate)

            # update input_ids, attention mask and length for the next step
            input_ids = torch.cat([input_ids,  next_token[:, None]], dim=-1)
            # update model kwargs
            model_kwargs = self.update_model_kwargs(model_kwargs=model_kwargs, presents=presents)
            # update sequence length
            cur_len += 1

            # if eos_token was found in one sentence, set sentence to finished
            binary_mask = (next_token != self.config.eos_token_id).long()  # of shape [batch_size]
            all_sequences_to_generate = all_sequences_to_generate.mul(binary_mask)  # of shape [batch_size]

            # stop when all sentences are finished or if we exceed the maximum length
            if all_sequences_to_generate.max() == 0 or ( cur_len >= max_length):
                break

        return input_ids
    
    @staticmethod
    def reorder_past_layer(
        layer_past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Reorders the past layer cache for beam search.

        Args:
            layer_past (Tuple[Tuple[torch.Tensor]]): Past layers.
            beam_idx (torch.Tensor): Indices for beam search.

        Returns:
            Tuple[Tuple[torch.Tensor]]: Reordered past layers.
        """
        """
        This function is used to re-order the `past layer` cache if [`beam_search`] .
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in layer_past
        )

    def beam_search(self, max_length: int = 300, image_hidden_states: Tensor = None, beam_size = 2,
                     device: device = None,debug: bool = False) -> Tensor:
        '''
        beam searching algorithm
        image_hidden_states: (batch_size,seq_len,hidden_size) -> (1,1,1024)
        beam_size: 2
        '''
        batch_size = image_hidden_states.size(0)
        seq_len = 1
        total_size = batch_size*beam_size

        input_ids = torch.full(size=(total_size, 1), fill_value=self.config.bos_token_id, dtype=torch.int64, device=device)
        attention_mask = torch.ones(size=(total_size, 1), dtype=torch.int64, device=device)
        model_kwargs = {"attention_mask": attention_mask,
                        "use_cache": True}
        
        # convert image_hidden_state from batch_size to total_size by copying the same hidden state
        # image_hidden_states = image_hidden_states.repeat(1,beam_size,1).view(total_size,-1)
        

        beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=beam_size,
                device=device,
                length_penalty=1.0,  # length_penalty > 0.0 encourages the model to generate shorter sequences
                do_early_stopping=False,
                num_beam_hyps_to_keep=1,
            )
        
        # initialize beam_scores which stores the score of each token in the beam
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device= device) # (batch_size, beam_size)
        beam_scores[:, 1:] = -1e9 # setting the score of all tokens except the first one to -inf
        beam_scores = beam_scores.view((batch_size * beam_size,)) # (batch_size * beam_size,)

        # beam search
        while True:
            # prepare model inputs (attention mask, layer_past, inputs_ids, position_ids, use_cache)
            model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids, seq_len= seq_len, **model_kwargs)
            
            # forward pass to get next
            logits, presents = self.forward(**model_inputs, image_hidden_states=image_hidden_states)
            logits = logits[:, -1, :]  # of shape [batch_size x vocab_size]
            
            # calculate probabilities of logits
            next_token_scores = nn.functional.log_softmax(logits, dim=-1)  # (batch_size * beam_size, vocab_size)
        
            # add beam_scores of previous sentences to all probabilities of tokens
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores) # (batch_size * beam_size , vocab_size)

            # reshape next_token_scores to (batch_size, beam_size * vocab_size)
            next_token_scores = next_token_scores.view((batch_size, beam_size * self.config.vocab_size)) # (batch_size, beam_size * vocab_size)

            # select top-k tokens
            next_token_scores, next_tokens = torch.topk(next_token_scores, k=beam_size*2, dim=1,largest=True, sorted=True) # (batch_size, beam_size*2) , (batch_size, beam_size*2)
            # get indices of top-k tokens
            # beam_idx = next_tokens 
            # beam_idx = next_tokens // self.config.vocab_size
                
            next_indices = torch.div(next_tokens, self.config.vocab_size, rounding_mode="floor") # (batch_size, beam_size*2)
            next_tokens = next_tokens % self.config.vocab_size # (batch_size, beam_size*2)           

            # calculate beam_scores
            beam_outputs = beam_scorer.process(
                input_ids, next_token_scores, next_tokens, next_indices, pad_token_id=self.config.pad_token_id,eos_token_id=self.config.eos_token_id
            )
            beam_scores = beam_outputs["next_beam_scores"] # (batch_size * beam_size,)
            beam_next_tokens = beam_outputs["next_beam_tokens"] # (batch_size * beam_size,)
            beam_idx = beam_outputs["next_beam_indices"] # (batch_size * beam_size,)

            # update input_ids, attention mask and length for the next step
            # beam_next_tokens = beam_next_tokens.view((batch_size * beam_size, 1))

            # input_ids = torch.cat([input_ids, beam_next_tokens], dim=-1)
            #TODO: test if the following line is necessary
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self.update_model_kwargs(model_kwargs=model_kwargs, presents=presents)

            # check if there is past layer
            if model_kwargs["layer_past"] is not None:
                
                # reorder past layer
                model_kwargs["layer_past"] = self.reorder_past_layer(
                    layer_past=model_kwargs["layer_past"], beam_idx=beam_idx
                )
                
            seq_len +=1
            # if all sentences are finished or if we exceed the maximum length
            if beam_scorer.is_done or (seq_len >= max_length):
                break


        # print("input_ids length: ",input_ids.size())
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            max_length=max_length+1,
        )
        # print("sequence_outputs: ",sequence_outputs)

        # delete unused variables
        # del beam_scores
        # del beam_outputs
        # del beam_next_tokens
        # del beam_idx
        # del next_tokens
        # del next_indices
        # del next_token_scores
        # del logits
        # del presents
        # del model_inputs
        # del input_ids
        # del attention_mask
        # del model_kwargs
        # del beam_scorer
        # del beam_size
        # del batch_size
        # gc.collect()
        return  sequence_outputs["sequences"]
        # return input_ids

if __name__ == "__main__":
    config = Config()
    # load small gpt2 config
    config.d_model = 768
    config.d_ff1 = 768
    config.d_ff2 = 768
    config.d_ff3 = 768
    config.num_heads = 12
    config.num_layers = 12
    config.vocab_size = 50257
    config.pretrained_model = "gpt2"
    config.max_seq_len = 1024
    config.ignore_index = -100
    image_config = Config()
    image_config.d_model = 1024
    image_config.d_ff1 = 1024
    image_config.d_ff2 = 768
    image_config.d_ff3 = 768
    image_config.num_heads = 16
    image_config.num_layers = 24
    image_config.vocab_size = 50257
    model = CustomGPT2(config,image_config)
    model.train()
    # model.convert_to_half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    summary(model, input_size=(1, 1, 768), device="cpu")

    # test the model full forward pass and backward pass
    batch_size = 1
    seq_len = 1024
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    image_hidden_states = torch.randn((batch_size, 1, 1024))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    image_hidden_states = image_hidden_states.to(device)
    labels = labels.to(device)
    for i in range(10):
        print(f"Running forward and backward pass {i+1}")
        output = model(
        input_ids = input_ids,
        layer_past = None,
        attention_mask = attention_mask,
        position_ids = None,
        inputs_embeds = None,
        image_hidden_states = image_hidden_states,
        labels = labels,
        use_cache = None,
        output_attentions = None,
        seq_len = None
        )
        # print(output)
        loss = output[0]
        loss.backward()
        print("Model forward and backward pass successful")
        print("All tests passed!")

    

