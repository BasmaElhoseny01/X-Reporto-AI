from matplotlib import cm, pyplot as plt
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

                if DEBUG == 0:
                    print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(self.data_loader_train)}] Loss: {losses.item()}")
                    break

            print(f"Epoch [{epoch}/{EPOCHS}] Loss: {epoch_losses/len(self.data_loader_train)}")
            
            if epoch%5==0 and epoch!=0:
                self.lr_scheduler.step()
                # Save the model
                torch.save(self.model.state_dict(), 'models/'+str(RUN)+'/heat_map.pth')

    def test(self):
        print("Testing")
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(self.data_loader_test):
            # Move to device
            images = images.to(DEVICE)
            targets=targets.to(DEVICE)
            targets=targets.type(torch.float32)


            with torch.no_grad():
                # Forward Pass
                feature_map,y=self.model(images)

                if targets is not None:
                    # Compute Loss
                    losses = self.criterion(y, targets)
                    print(f"Test Batch [{batch_idx}/{len(self.data_loader_test)}] Loss: {losses.item()}")
                y=y.to('cpu')
                # get index where prediction is 1
                # y=y>0.5
                # print(y)
                # classes = np.where(y[0] == 1)[0]
                self.generate_heat_map(feature_map,image=images[0],classes=y[0])
                break
    
    def generate_heat_map(self,feature_map,image,classes):
        b, c, h, w = feature_map.size()
        # print(feature_map.shape)  # torch.Size([batch_size, 2048, 16, 16])

        # Reshape feature map
        feature_map = feature_map.view(b, c, h*w).transpose(1, 2)
        # print(feature_map.shape) # torch.Size([1, 256, 2048])
        fc_weight=torch.nn.Parameter(self.model.fc.weight.t().unsqueeze(0))
        # print(fc_weight.shape) # torch.Size([1, 2048, 14])

        # Perform batch matrix multiplication
        cam = torch.bmm(feature_map, fc_weight).transpose(1, 2) #torch.Size([1, 14, 256])

        ## normalize to 0 ~ 1
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val

        cam = cam.view(b, -1, h, w)
        cam = torch.nn.functional.interpolate(cam, 
                                (image.size(1), image.size(2)), mode='bilinear', align_corners=True).squeeze(0)
        # print(cam.shape) # torch.Size([13, 512, 512])
       
        # multiply each heatmap by the corresponding class probability
        for i in range(cam.shape[0]):
            cam[i] = cam[i] * classes[i]
        # cam = torch.stack(cam)
            
        # print(cam.shape) # torch.Size([13, 512, 512])
        cam = torch.split(cam, 1)
        # print(cam.shape) #torch.Size([1, 512, 512])


        # image = image.permute(1, 2, 0)
        # # normalize the image
        # image = (image - image.min()) / (image.max() - image.min())
        # image=image.to('cpu')
        # plt.imshow(image, cmap='hot')
        # plt.show()

        image=imshow(image)
        heatmapList = []
        # Load the heatmap images into a list
        # Convert images to NumPy arrays
        img_np = np.array(image)        
        # print(classes)
        for k in range(len(classes)):
            cam_ = cam[k].squeeze().cpu().data.numpy()
           
            # displaied as infera red image 
            # cam_pil = Image.fromarray(np.uint8(matplotlib.cm.hot(cam_)*255)).convert("RGB")
           
            # displaied black and red for important area
            # cam_pil = Image.fromarray(np.uint8(matplotlib.cm.afmhot(cam_)*255)).convert("RGB")
           
            # colored but slightly blue
            cam_pil = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(cam_)*255)).convert("RGB")
            heatmapList.append(np.array(cam_pil) )

        # Calculate the blended image
        alpha = 0.25  # Alpha value for blending (adjust as needed)
        blended_image_np = img_np.copy().astype(np.float32)
        for cam_np in heatmapList:
            blended_image_np += alpha * cam_np
        # Clip the pixel values to the valid range [0, 255]
        blended_image_np = np.clip(blended_image_np, 0, 255).astype(np.uint8)
        # Convert the resulting array back to a PIL image
        blended_image_pil = Image.fromarray(blended_image_np).convert("RGB")
        # Display or save the blended image in rgb formate
        blended_image_pil.show()          

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

    heat_map_model=HeatMap().to(DEVICE)

    # Freezing
    # for name, param in heat_map_model.named_parameters():
    #     param.requires_grad = False
    # summary(heat_map_model, input_size=(4, 3, 512, 512))

    # trainer = Heat_Map_trainer(model=None,training_csv_path=Heat_map_train_csv_path)
    
    trainer = Heat_Map_trainer(model=heat_map_model,training_csv_path=Heat_map_train_csv_path)
    trainer.train()
        
    # Testing
    trainer.test()