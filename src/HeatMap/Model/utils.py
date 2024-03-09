import math
import torch
from torch import nn

def pos_encode(d_model,height,width):
    """
    d_model: dimension of the model
    height: height of the positions
    width: width of the positions
    return d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = d_model // 2
    # Compute the position
    div_term = torch.exp(torch.arange(0., d_model, 2)* -(math.log(10000.0) / d_model))
    # Compute the position
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    # Compute the sin and cos
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).expand(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).expand(1, 1,width)
    pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(1).expand(1, 1,width)
    return pe

def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def custom_replace(tensor,on_neg_1,on_zero,on_one):
    res = tensor.clone()
    res[tensor==-1] = on_neg_1
    res[tensor==0] = on_zero
    res[tensor==1] = on_one
    return res