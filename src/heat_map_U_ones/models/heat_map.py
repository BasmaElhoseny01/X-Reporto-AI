from torch import nn
import torch
from torchvision.models import ResNet50_Weights
from torchvision import models

import torch.nn.functional as F

from torchsummary import summary
from config import *


class HeatMap(nn.Module):
    def __init__(self):
        super(HeatMap, self).__init__()
        self.model = models.densenet121()
        # [Fix] The Paper is 13
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 13)
        
    def forward(self, x):
        x=self.model.features(x)
        
        # Apply Global Average Pooling to feature maps
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Concatenate the two tensors
        x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.model.classifier(x)
        
        # Apply sigmoid activation
        # scores = torch.sigmoid(x)
        scores=x

        return x,scores



# from torchinfo import summary


# model= HeatMap().to('cuda')
# # print(model)
# summary(model, input_size=(4, 3, 224, 224) )


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