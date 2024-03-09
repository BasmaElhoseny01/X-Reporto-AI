import numpy as np
import torch
from torch import nn
from src.HeatMap.Model.Backbone import Backbone
from src.HeatMap.Model.utils import custom_replace, pos_encode, weights_init
from src.HeatMap.Model.SelfAttention import SelfAttention


class HeatMap(nn.Module):
    def __init__(self,num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False):
        super(HeatMap, self).__init__()
        #  use label masking technique or not
        self.use_lmt=use_lmt
        self.no_x_features=no_x_features
        self.backbone=Backbone()
        hidden=2048 # 2048 for resnet50 output features

        # Label Embeddings
        # initialize the label embeddings as a trainable parameter in the model. 
        # Each label is represented by a vector of size hidden (2048 in this case), 
        # and these embeddings will be learned during the training process.
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)
        
        # State Embeddings
        # 3 state embeddings for pos, neg and unknown labels
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Positional Encoding
        self.pos_emb=pos_emb
        if self.pos_emb:
            self.pos_encoder = pos_encode(hidden, 18, 18).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttention(hidden,heads,dropout) for _ in range(layers)])

        # Output Layer
        self.output_linear=nn.Linear(hidden,num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images,mask):
        const_label_input = self.label_input.repeat(images.size(0),1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        features = self.backbone(images)
        if self.pos_emb:
            pos_encoding = self.pos_encoder(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            features = features + pos_encoding
            features = features.view(features.size(0),features.size(1),-1).permute(0,2,1) 

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask,0,1,2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features,init_label_embeddings),1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        output = self.output_linear(label_embeddings) 
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).cuda()
        output = (output*diag_mask).sum(-1)

        return output,None,attns

