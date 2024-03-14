from torch import nn
import torch
from torchvision.models import ResNet50_Weights
from torchvision import models

import torch.nn.functional as F

from torchsummary import summary
from config import *


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        # Perform global average pooling along the spatial dimensions (height and width)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# Example usage:
# # Create an instance of the GlobalAveragePooling layer
# gap_layer = GlobalAveragePooling()


class HeatMap(nn.Module):
    def __init__(self):
        super(HeatMap, self).__init__()
        self.feature_Layers=models.densenet121(pretrained=True)
        num_ftrs = self.feature_Layers.classifier.in_features
        self.feature_Layers.classifier = nn.Linear(num_ftrs, 13)
        
    def forward(self, x):
        y=self.feature_Layers(x) 
        # y=F.sigmoid(y) #sigmoid as we use BCELoss
        return y



# from torchinfo import summary


# model= HeatMap().to('cuda')
# print(model)
# summary(model, input_size=(4, 3, 512, 512) )


# Freezing
# for name, param in model.named_parameters():
#     param.requires_grad = False
# summary(model, input_size=(4, 3, 512, 512))

###########################################################################

# # pip install torchsummary


# You need to define input size to calcualte parameters
# torchsummary.summary(model,batch_size=4, input_size=(3, 512, 512))


# print("Demo of Data Flow")
# input_data = torch.randn(4,3,512, 512).to('cuda')
# output=model.to('cuda')(input_data)
# # Apply softmax activation
# print(output.shape)
    
# from torchvision import models
# from torchsummary import summary

# vgg = models.vgg16()
# print(vgg)
# summary(vgg, (3, 512, 512),device='cpu')
# summary(vgg, input_size=(4, 3, 512, 512))