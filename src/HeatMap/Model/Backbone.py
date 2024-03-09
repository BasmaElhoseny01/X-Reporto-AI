from torch import nn
import torch
from torchinfo import summary
from torchvision.models import ResNet50_Weights
from torchvision import models
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.base_network=models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last two layers
        self.base_network=nn.Sequential(*list(self.base_network.children())[:-2])
        # self.base_network.avgpool = nn.AvgPool2d(kernel_size=7,stride=1,padding=0)

        
    def forward(self,images):
        x = self.base_network(images)
        # x = self.base_network.avgpool(x)
    
        return x        
    
# Example usage:
# model= Backbone().to('cuda')
# print(model)
# summary(model, input_size=(4, 3, 512, 512))