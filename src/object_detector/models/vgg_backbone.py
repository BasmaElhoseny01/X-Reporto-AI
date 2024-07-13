import torchvision
import torch
import torch.nn as nn

model = torchvision.models.vgg16(pretrained=True)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
model.features[28] = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
fe = list(model.features)
dummy_img = torch.zeros((1, 1, 512, 512)).float()

req_features = []
k = dummy_img.clone()
for i in fe:
    k = i(k)
    if k.size()[2] < 512//16:
        req_features.append(i)
        break
    req_features.append(i)
    out_channels = k.size()[1]

faster_rcnn_fe_extractor = nn.Sequential(*req_features)
out_map = faster_rcnn_fe_extractor(dummy_img)
