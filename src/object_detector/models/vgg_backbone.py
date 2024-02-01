import torchvision
import torch
import torch.nn as nn

model = torchvision.models.vgg16(pretrained=True)
# print(model)
# print(fe) # length is 15
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
model.features[28] = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
fe = list(model.features)
print(model)
dummy_img = torch.zeros((1, 1, 512, 512)).float()

req_features = []
k = dummy_img.clone()
for i in fe:
    k = i(k)
    if k.size()[2] < 512//16:
        print("i: ",i)
        print("k.size(): ",k.size())
        req_features.append(i)
        break
    req_features.append(i)
    out_channels = k.size()[1]
print(len(req_features)) #30
print(out_channels) # 512

faster_rcnn_fe_extractor = nn.Sequential(*req_features)
out_map = faster_rcnn_fe_extractor(dummy_img)
print("out_map: ",out_map.size()) # torch.Size([1, 2048, 16, 16])