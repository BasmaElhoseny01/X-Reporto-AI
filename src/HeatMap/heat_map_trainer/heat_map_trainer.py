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
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss  -(y log(p)+(1-y)log(1-p))    

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)

        # TODO transform_type->Train
        self.dataset_train = HeatMapDataset(dataset_path= training_csv_path, transform_type='train')
        self.dataset_test = HeatMapDataset(dataset_path= testing_csv_path, transform_type='val')
        
        # create data loader
        # TODO suffle Training Loaders
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        self.data_loader_test = DataLoader(dataset=self.dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print(" self.dataset_train",len(self.dataset_train))
        print(" self.dataset_test",len(self.dataset_test))


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

                losses= 0
                # Loss as summation of Binary cross entropy for each class :D
                # Only 13 Classes becuase we removed the class of nofinding 
                for c in range(13):
                  losses+=self.criterion(y[:,c],targets[:,c])

                # apply threshold to make it binary
                # y=y>0.5
                # y=y.type(torch.float32)
                # y.requires_grad=True

                # # Compute Loss
                # losses=0
                # # calc loss for each label
                # for i in range (y.shape[0]):
                #     for j in range(y.shape[1]):
                #         losses += self.criterion(y[i][j], targets[i][j])
                # sys.exit()
                # # loss = 0
                # # for c in range(14):
                #     # loss += F.binary_cross_entropy(predicted[:, c], target[:, c], reduction='mean')
                        
                # Backward Pass
                losses.backward()

                # Update the weights
                self.optimizer.step()

                # Add losses
                epoch_losses+=losses

                if DEBUG == 0:
                    print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(self.data_loader_train)}] Loss: {losses.item()}")
                    # break

            print(f"Epoch [{epoch}/{EPOCHS}] Loss: {epoch_losses/len(self.data_loader_train)}")
            
            if epoch%5==0 and epoch!=0:
                self.lr_scheduler.step()
                # Save the model
                torch.save(self.model.state_dict(), 'models/'+str(RUN)+'/heat_map.pth')

    def test(self):
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(self.data_loader_test):
            # Move to device
            images = images.to(DEVICE)
            targets=targets.to(DEVICE)
            targets=targets.type(torch.float32)


            with torch.no_grad():
                # Forward Pass
                feature_map,y=self.model(images)
                print("y",y)
                print("targets",targets)
                # print("y.shape",y.shape)
                # print("feature_map",feature_map.shape)

                # probabilit=y[0]
                # probabilit=probabilit.to('cpu')

                if targets is not None:
                    # Computing Losses For the Batch
                    losses= 0
                    # Loss as summation of Binary cross entropy for each class :D
                    # Only 13 Classes becuase we removed the class of nofinding 
                    for c in range(13):
                      losses+=self.criterion(y[:,c],targets[:,c])

                    print(f"Test Batch [{batch_idx}/{len(self.data_loader_test)}] Loss: {losses.item()}")

                # Generating Heat Map
                prob=y
                y=(y>0.5)*1.0
                self.generate_heat_map(feature_map,image=images,classes=y,prob=prob)

                # classes = np.where(y[0] == 1)[0]
                # print("classes",classes)

                # y=y.type(torch.float32)
                # y=y>0.5
                # print(probabilit)
                # print(y)
                # # get index where prediction is 1
                # # y=y>0.5
                # # # print(y)
                # classes = np.where(y[0] == 1)[0]
                # self.generate_heat_map(feature_map,image=images[0],classes=probabilit)
                break
    
    def generate_heat_map(self,feature_map,image,classes,prob):
        '''
        Heat Map for 1 Image
        '''
        # print("generate_heat_map")
        # print("feature_map",feature_map.shape)
        # print("classes",classes)
        # print("classes.shape",classes.shape)
        # print("torch.nn.Parameter(self.model.fc.weight.t())",torch.nn.Parameter(self.model.fc.weight.t().unsqueeze(0)).shape)
        # print("torch.nn.Parameter(self.model.fc.weight.t())",torch.nn.Parameter(self.model.fc.weight.t().unsqueeze(0)))
        print("image.shape",image.shape)
       
        # Generate class activation map
        b, c, h, w = feature_map.size()  #1,1024,16,16
        feature_map = feature_map.view(b, c, h*w).transpose(1, 2)
        print("feature_map.shape",feature_map.shape)  # torch.Size([batch_size, 256, 1024])

        # Classification Layer Weight 
        fc_weight=torch.nn.Parameter(self.model.fc.weight.t().unsqueeze(0)) #(D*C)
        print("fc_weight.shape",fc_weight.shape) # torch.Size([1, 2048, 13])

        # Batch Matrix Multiplication
        cam = torch.bmm(feature_map, fc_weight).transpose(1, 2) #torch.Size([1, 13, 256])
        print("cam.shape",cam.shape) #torch.Size([1, 13, 256])

        ## normalize to 0 ~ 1
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val

        # Fix Dimension
        cam = cam.view(b, -1, h, w)
        print("cam.shape",cam.shape) #torch.Size([1,13,16,16])


        # Interpolation
        cam = nn.functional.interpolate(cam, 
                                        (image.size(2), image.size(3)), mode='bilinear', align_corners=True).squeeze(0)
        print("cam.shape",cam.shape)
        
        # Split By classes
        cam = torch.split(cam, 1)
        print("len(cam)",len(cam))
        print("cam[0].shape",cam[0].shape)
        

        # tensor to pil image
        img_pil = imshow(image)
        img_pil.save('/content/'+"input.jpg")


        for k in range(13):
            print("Predict '%s' with %2.4f probability"%(k, prob[0][k]))
            cam_ = cam[k].squeeze().cpu().data.numpy()
            cam_pil = Image.fromarray(np.uint8(cm.gist_earth(cam_)*255)).convert("RGB")
            cam_pil.save('/content/'+"cam_class__%s_prob__%2.4f.jpg"%(k, prob[0][k]))

            # overlay image and class activation map
            blended_cam = Image.blend(img_pil, cam_pil, 0.2)
            blended_cam.save('/content/'+"blended_class__%s_prob__%2.4f.jpg"%(k,prob[0][k]))
    

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

    trainer = Heat_Map_trainer(model=None,training_csv_path=Heat_map_train_csv_path)
    
    # trainer = Heat_Map_trainer(model=heat_map_model,training_csv_path=Heat_map_train_csv_path)
    # trainer.train()
        
    # Testing
    trainer.test()