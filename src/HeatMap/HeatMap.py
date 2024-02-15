from torch import nn
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        # Perform global average pooling along the spatial dimensions (height and width)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# Example usage:
# Create an instance of the GlobalAveragePooling layer
gap_layer = GlobalAveragePooling()


class HeatMapCAM(nn.Module):
    def __init__(self):
        super(HeatMapCAM, self).__init__()
        netlist = list(resnet50(pretrained=True).children())
        self.feature_extractor = nn.Sequential(*netlist[:-2])
        self.fc = netlist[-1]
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        self.fc_weight =  nn.Parameter(self.fc.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        prob, args = torch.sort(output, dim=1, descending=True)
        
        # get all class with value =1 
        topk_arg = args[:, :1]
        
        # generage class activation map
        b, c, h, w = feature_map.size()
        feature_map = feature_map.view(b, c, h*w).transpose(1, 2)

        cam = torch.bmm(feature_map, self.network.fc_weight).transpose(1, 2)

        ## normalize to 0 ~ 1
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val
        
        ## top k class activation map
        topk_cam = cam.view(1, -1, h, w)[0, topk_arg]
        topk_cam = nn.functional.interpolate(topk_cam.unsqueeze(0), 
                                        (x.size(2), x.size(3)), mode='bilinear', align_corners=True).squeeze(0)
        topk_cam = torch.split(topk_cam, 1)

        return topk_arg, topk_cam


x=HeatMapCAM()
print(x)


