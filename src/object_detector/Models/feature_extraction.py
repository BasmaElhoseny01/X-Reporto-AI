import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator,RPNHead,RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor


class FeatureNetwork:
    def __init__(self,model_type):
        self.model_type = model_type
        self.model = None
        self.feature_extractor = None
        self.out_channels = None
        self.image_size = None
        self.device = None
        self.model = None
        self.feature_extractor = None

        if model_type == "vgg16":
            self.model = torchvision.models.vgg16(pretrained=True)
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            self.model.features[28] = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
            self.feature_extractor = self.model.features
            self. feature_extractor = nn.Sequential(*list(feature_extractor.children()))
            self.out_channels = 2048
            self.image_size = 512
        elif model_type == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])
            self.feature_extractor.out_channels = self.model.layer4[-1].conv3.out_channels
            self.out_channels = self.feature_extractor.out_channels
            self.image_size = 512
    
    def __call__(self, x):
        return self.feature_extractor(x)


