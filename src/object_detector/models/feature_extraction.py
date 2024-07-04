import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator,RPNHead,RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor


class FeatureNetwork(nn.Module):
    def __init__(self,model_type):
        super().__init__()
        self.model_type = model_type
        self.model = None
        self.feature_extractor = None
        self.out_channels = None
        self.image_size = None
        self.device = None
        self.model = None
        self.feature_extractor = None

        if model_type == "vgg16":
            self.feature_extractor = torchvision.models.vgg16(pretrained=True)
            self.feature_extractor.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            self.feature_extractor.features[28] = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
            self.feature_extractor = self.feature_extractor.features
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children()))
            self.out_channels = 2048
            self.image_size = 512
        elif model_type == "resnet50":
            self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_extractor.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.out_channels = self.feature_extractor.layer4[-1].conv3.out_channels
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
            # self.feature_extractor.out_channels = self.feature_extractor.layer4[-1].conv3.out_channels
            self.image_size = 512
    
    def __call__(self, x):
        return self.feature_extractor(x)
