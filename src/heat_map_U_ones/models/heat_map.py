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
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 14)
        
    def forward(self, x):
        features=self.model.features(x)
        
        # Apply Global Average Pooling to feature maps
        y_pred = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Concatenate the two tensors
        y_pred = torch.flatten(y_pred, 1)
        y_pred = y_pred.view(y_pred.size(0), -1)
        
        # Classifier
        y_pred = self.model.classifier(y_pred)
        
        # Apply Sigmoid
        y_scores=torch.sigmoid(y_pred)

        return y_pred,y_scores,features



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