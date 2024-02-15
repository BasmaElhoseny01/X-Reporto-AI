from torchinfo import summary
import torch.optim as optim
import torchvision
from PIL import Image
import numpy as np
import matplotlib

from src.HeatMap.DataLoader.Dataset import HeatMapDataset
from torch.utils.data import  DataLoader

from src.HeatMap.HeatMap import HeatMap

from config import *

class Heat_Map_trainer:
    def __init__(self,model,training_csv_path='datasets/train.csv'):
        '''
        inputs:
            training_csv_path: string => the path to the training csv file
            model: the object detector model
        '''
        self.model=model

        # Move to device
        self.model.to(DEVICE)
        
        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= LEARNING_RATE)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)


        self.dataset_train = HeatMapDataset(dataset_path= training_csv_path, transform_type='val')
        print("Dataset Loaded")
        
        # create data loader
        # TODO suffle Training Loaders
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print("DataLoader Loaded")


    def train(self):
        self.model.train()
        for epoch in range(EPOCHS):
            for batch_idx, (images, targets) in enumerate(self.data_loader_train):
                targets[0][:]=torch.tensor([True, False, True, False, True, False, True, False, True, False, True, False, True, False])

                # add new dimension to images after batch size
                images = torch.stack([image.to(DEVICE) for image in images])

                # targetdata=[]
                # for i in range(len(images)):
                #     targets[i]=targets[i].to(DEVICE)
                #     targetdata.append(targets[i])
            
                # print("targetdata.shape",len(targetdata))

                feature_map,y=self.model(images)
                print(feature_map.shape)
                print(y)

                # Generate HeatMap
                # generage class activation map
                b, c, h, w = feature_map.size()
                feature_map = feature_map.view(b, c, h*w).transpose(1, 2)

                fc_weight=torch.nn.Parameter(self.model.fc.weight.t().unsqueeze(0))

                print(feature_map.shape) # torch.Size([1, 256, 1024])
                print(fc_weight.shape) # torch.Size([1, 1024, 14])

                # Perform batch matrix multiplication
                cam = torch.bmm(feature_map, fc_weight).transpose(1, 2) #torch.Size([1, 14, 256])
                print(cam.shape)

                ## normalize to 0 ~ 1
                min_val, min_args = torch.min(cam, dim=2, keepdim=True)
                cam -= min_val
                max_val, max_args = torch.max(cam, dim=2, keepdim=True)
                cam /= max_val


                # prob, args = torch.sort(y, dim=1, descending=True)
                # topk_args=args.squeeze().tolist()[:10]
                # print(topk_args)

                topk_cam = cam.view(b, -1, h, w)
                print(topk_cam)
                print(topk_cam.shape)#torch.Size([1, 14, 16, 16]) torch.Size([10, 16, 16])

                topk_cam = torch.nn.functional.interpolate(topk_cam, 
                                        (images.size(2), images.size(3)), mode='bilinear', align_corners=True).squeeze(0)
                # print(topk_cam)
                print(topk_cam.shape) # torch.Size([14, 512, 512])

                topk_cam = torch.split(topk_cam, 1)
                print(len(topk_cam)) #torch.Size([1, 512, 512])
                print(topk_cam[0].shape) #torch.Size([1, 512, 512])


                # Show Image
                img_pil = imshow(images[0])
                img_pil.save("./input.jpg")

                # Show CAM
                for k in range(10):
                    print(topk_cam[k].shape) #torch.Size([1, 512, 512])
                    cam_ = topk_cam[k].squeeze().cpu().data.numpy()
                    print(cam_)

                    cam_pil = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(cam_)*255)).convert("RGB")
                    cam_pil.save("./cam_class"+str(k)+'.jpg')


                    # overlay image and class activation map
                    blended_cam = Image.blend(img_pil, cam_pil, alpha=0.25)
                    blended_cam.save("./blended_class__"+str(k)+'.jpg')

                    sys.exit()

def imshow(tensor):
    denormalize = _normalizer(denormalize=True)    
    if tensor.is_cuda:
        tensor = tensor.cpu()    
    tensor = torchvision.utils.make_grid(denormalize(tensor.squeeze()))
    image = torchvision.transforms.functional.to_pil_image(tensor)
    return image


def _normalizer(denormalize=False):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]    
    
    if denormalize:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]
    
    return torchvision.transforms.Normalize(mean=MEAN, std=STD)

if __name__ == '__main__':

    heat_map_model=HeatMap().to('cuda')

    # Freezing
    for name, param in heat_map_model.named_parameters():
        param.requires_grad = False
    summary(heat_map_model, input_size=(4, 3, 512, 512))

    trainer = Heat_Map_trainer(model=heat_map_model,training_csv_path=Heat_map_train_csv_path)
    trainer.train()