import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from torchsummary import summary

# from torchvision.models import ResNet50_Weights

import sys
import numpy as np

from config import *

class HeatMap(nn.Module):
    def __init__(self):
        '''
        Initializes the HeatMap model
        '''
        super(HeatMap, self).__init__()

        # DenseNet-121 as backbone
        self.model = models.densenet121(pretrained=False)

        # Modify the classifier to output 8 classes for the Classes in the dataset
        self.model.classifier = nn.Linear(self.model.classifier.in_features, len(CLASSES))
        
        # Optimal Thresholds
        self.optimal_thresholds=[]

        # Gradients
        self.gradients=None

    def get_gradients(self):
        '''
        Returns the gradients of the model
        '''
        return self.gradients

    # def activations_hook(self, grad):
    #     ''''''
    #     self.gradients = grad
    
    def forward(self, x):
        '''
        Forward pass of the model
        '''
        # Features extraction from model backbone
        features=self.model.features(x) # shape: (batch_size, 1024, 7, 7)
        
        # if has_nan:
        #     print("Input data contains NaN values.")
        # else:
            # print("Input data does not contain NaN values.")
        
        # Register the hook
        # features.register_hook(self.activations_hook)

        # Apply Global Average Pooling to feature maps (Adaptive Average Pooling) (H, W) -> (1, 1) [Similar to GAP in CNNs]
        y_pred = F.adaptive_avg_pool2d(features, (1, 1)) # shape: (batch_size, 1024, 1, 1)
        
        # Concatenate the two tensors
        y_pred = torch.flatten(y_pred, 1) # shape: (batch_size, 1024)
        y_pred = y_pred.view(y_pred.size(0), -1) # shape: (batch_size, 1024)
        
        # Classifier
        y_pred = self.model.classifier(y_pred) # shape: (batch_size, num_classes)
        
        # Apply Sigmoid activation
        y_scores=torch.sigmoid(y_pred)  # shape: (batch_size, num_classes)

        return y_pred,y_scores,features

    def __str__(self):
        '''
        Returns the string representation of the model to be printed with its optimal thresholds
        '''
        model_str = str(self.model)
        thresholds_str = str(self.optimal_thresholds)
        return f"Model:\n{model_str}\nOptimal Thresholds:\n{thresholds_str}"

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        '''
        Returns the state dictionary of the model with the optimal thresholds
        '''
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict['optimal_thresholds'] = self.optimal_thresholds
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        '''
        Loads the state dictionary of the model with the optimal thresholds
        '''
        self.optimal_thresholds = state_dict.pop('optimal_thresholds', [])
        super().load_state_dict(state_dict, strict)





############################################################################################################################################################
############################################################# Model Arch & Layer Testing ###################################################################
############################################################################################################################################################
# from torchinfo import summary
# model= HeatMap().to('cuda')
# # print(model)

# print("Model Summary")
# summary(model, input_size=(4, 3, 224, 224) )

# # run forwad pass
# print("Demo of Data Flow")
# input_data = torch.randn(4,3,224, 224).to('cuda')
# output=model.to('cuda')(input_data)
# Apply softmax activation


# Freezing
# for name, param in model.named_parameters():
#     param.requires_grad = False
# summary(model, input_size=(4, 3, 512, 512))

###########################################################################

# # pip install torchsummary

# import torchsummary

# model= HeatMap().to('cuda')
# # You need to define input size to calcualte parameters
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