from torch import nn

from torchinfo import summary
import torch.optim as optim
import torchvision
from PIL import Image
import numpy as np
import matplotlib

from src.HeatMap.DataLoader.Dataset import HeatMapDataset
from torch.utils.data import  DataLoader

from src.HeatMap.HeatMap import HeatMap

from src.HeatMap.config import *

class Heat_Map_trainer:
    def __init__(self,model=None,training_csv_path=Heat_map_train_csv_path,testing_csv_path=Heat_map_test_csv_path):
        '''
        inputs:
            training_csv_path: string => the path to the training csv file
            model: the object detector model
        '''
        self.model=model
        if CONTINUE_TRAIN or model is None:
            self.model=HeatMap().to(DEVICE)
            self.model.load_state_dict(torch.load('models/'+str(RUN)+'/heat_map.pth'))

        # Move to device
        self.model.to(DEVICE)
        
        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= LEARNING_RATE)


        # Create Criterion 
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)

        # TODO transform_type->Train
        self.dataset_train = HeatMapDataset(dataset_path= training_csv_path, transform_type='val')
        self.dataset_test = HeatMapDataset(dataset_path= testing_csv_path, transform_type='val')
        
        # create data loader
        # TODO suffle Training Loaders
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        self.data_loader_test = DataLoader(dataset=self.dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    def train(self):
        self.model.train()
        for epoch in range(EPOCHS):
            epoch_losses=0
            for batch_idx, (images, targets) in enumerate(self.data_loader_train):
                # Move to device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)
                # convert targets to float
                targets=targets.type(torch.float32)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward Pass
                feature_map,y=self.model(images)

                # Compute Loss
                losses = self.criterion(y, targets)

                # Backward Pass
                losses.backward()

                # Update the weights
                self.optimizer.step()

                # Add losses
                epoch_losses+=losses

                if DEBUG:
                    print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(self.data_loader_train)}] Loss: {losses.item()}")
                    break

            print(f"Epoch [{epoch}/{EPOCHS}] Loss: {epoch_losses/len(self.data_loader_train)}")
            
            if epoch%5==0:
                self.lr_scheduler.step()
                # Save the model
                torch.save(self.model.state_dict(), 'models/'+str(RUN)+'/heat_map.pth')

            continue
            # sys.exit()
        

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

    def test(self):
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(self.data_loader_test):
            # Move to device
            images = images.to(DEVICE)
            targets=targets.to(DEVICE)


            with torch.no_grad():
                # Forward Pass
                feature_map,y=self.model(images)

                if targets is not None:
                    # Compute Loss
                    losses = self.criterion(y, targets)
                    print(f"Test Batch [{batch_idx}/{len(self.data_loader_test)}] Loss: {losses.item()}")
            

                

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
    # for name, param in heat_map_model.named_parameters():
    #     param.requires_grad = False
    summary(heat_map_model, input_size=(4, 3, 512, 512))

    trainer = Heat_Map_trainer(model=heat_map_model,training_csv_path=Heat_map_train_csv_path)
    trainer.train()