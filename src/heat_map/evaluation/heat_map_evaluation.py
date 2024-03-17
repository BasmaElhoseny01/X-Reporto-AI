# Logging
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from logger_setup import setup_logging
import logging

from datetime import datetime

import os
import gc
import sys

# Torch
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter

# Torch
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import torchvision

from PIL import Image

# Modules
from src.heat_map.models.heat_map import HeatMap
from src.heat_map.data_loader.dataset import HeatMapDataset
from src.utils import load_model

# Utils 
from config import *
from src.utils import plot_heatmap


class HeatMapEvaluation():
    def __init__(self, model:HeatMap,evaluation_csv_path:str = heat_map_evaluation_csv_path,tensor_board_writer:SummaryWriter=None):
        '''
        X-Reporto Validation Class
        Args:
        model: X-Reporto Model
        evaluation_csv_path: Path to the validation csv file
        ''' 
        self.model=HeatMap().to(DEVICE)
        if CONTINUE_TRAIN:
            logging.info("Loading heat_map ....")
            load_model(model=self.model,name='heat_map_epoch_10')
            print("model loaded")
        else:
            self.model = model
        self.evaluation_csv_path = evaluation_csv_path
        self.tensor_board_writer=tensor_board_writer

        self.model.to(DEVICE)

        # self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss  -(y log(p)+(1-y)log(1-p))    
        pos = torch.tensor(POS_WEIGHTS)*5
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum',pos_weight=pos).to(DEVICE)
        
        self.data_loader_val = DataLoader(dataset=HeatMapDataset(self.evaluation_csv_path), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        logging.info("Evaluation dataset loaded")
        print("Evaluation dataset loaded")

            
    def compute_weighted_losses(self,targets,y):
#         def weighted_binary_cross_entropy(y_true, y_pred, positive_weight, negative_weight):
#             # Ensure inputs are within valid range
#             y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)

#             # Compute binary cross-entropy loss with weighted terms for positive and negative samples
#             loss = - (positive_weight * y_true * torch.log(y_pred) + negative_weight * (1 - y_true) * torch.log(1 - y_pred))

#             # Average the weighted loss across all samples
#             loss = torch.mean(loss)
            
#             return loss
        
        # Compute Losses
        Total_loss= 0
        # Loss as summation of Binary cross entropy for each class :D Only 13 Classes bec we removed the class of no-finding 
        for c in range(13):
            # Construct weights tensor
            weights = targets[:,c] * POS_WEIGHTS[c] + (1 - targets[:,c]) * (1-POS_WEIGHTS[c])
     
            # Initialize the weighted BCE loss
            Total_loss += nn.BCELoss(weight=weights)(y[:,c],targets[:,c])
        return Total_loss
            
    
    def evaluate(self):
        #Evaluate the model
        scores = self.evaluate_heat_map()

        # [Tensor Board] Update the Board by the scalers for that Run
        # self.update_tensor_board_score()
        

    def evaluate_heat_map(self):
        self.model.eval()
        with torch.no_grad():
            # validate the model
            logging.info("Evaluating the model")
            validation_total_loss=0
            labels=[]
            # make predictions tensor empty
            predictions=[]
            precisionSoftmax=[]

            for batch_idx,(images,targets) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)
                # convert targets to float
                targets=targets.type(torch.float32)
                labels.append(targets)

                # Forward Pass [TODO]
                features,Total_loss,classes=self.forward_pass(images,targets)
                # apply threshold to the classes
                classesSoftmax=F.sigmoid(classes)
                precisionSoftmax.append((classesSoftmax>0.5).type(torch.float32))
                classes=(classes>0).type(torch.float32) 
                predictions.append(classes)
                # [Tensor Board] Draw the HeatMap Predictions of this batch
                #TODO: uncomment
                # self.draw_tensor_board(batch_idx,images,features,classes)

                validation_total_loss+=Total_loss

            print(f"before : ")
            f1_score,precision,recall=self.F1_score(torch.cat(labels,0),torch.cat(predictions,0))
            print(f"after : ")
            f1_score,precision,recall=self.F1_score(torch.cat(labels,0),torch.cat(precisionSoftmax,0))
        # average validation_total_loss
        validation_total_loss/=(len(self.data_loader_val))
        print(f"Validation Loss: {validation_total_loss}")
        logging.info(f"Validation Loss: {validation_total_loss}")
        return validation_total_loss
    
    def F1_score(self, y_true, y_pred):
        '''
        F1 Score
        '''
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        false_positive = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        false_negative = np.sum(np.logical_and(y_true == 1, y_pred == 0))
        true_positive = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        true_negative = np.sum(np.logical_and(y_true == 0, y_pred == 0))
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')
        print(f'False Positive: {false_positive}, False Negative: {false_negative}, True Positive: {true_positive}, True Negative: {true_negative}')
        return f1,precision,recall
    
    def forward_pass(self,images,targets):
        Total_loss=0
        features=[]
        # Forward Pass
        y=self.model(images)
        # y=F.sigmoid(y) #sigmoid as we use BCELoss
        # features.append(feature_map)

        # Calculate Loss
        # Total_loss=self.compute_weighted_losses(targets=targets,y=y)
        Total_loss=self.criterion(y,targets)
        return features,Total_loss,y
    
    ########################################################### General Fuunctions ##########################################
    def update_tensor_board_score():
        pass

    def draw_tensor_board(self,batch_idx,images,features,classes):
        '''
        Add images to tensorboard
        '''
        images=images.to('cpu')
        # convert features and classes to tensor
        features=torch.stack(features)
        classes=torch.stack(classes)
        # print(features.shape) # torch.Size([1, 2, 1024, 16, 16]).
        # print(classes.shape) # torch.Size([1, 2, 13])
        classes=classes.squeeze(0)
        features=features.squeeze(0)
        for i in range(images.shape[0]):
            # Generate HeatMap
            heat_map=self.generate_heat_map(feature_map=features[i],image=images[i],classes=classes[i])
            # Add to Tensor Board
            self.tensor_board_writer.add_image(f'HeatMap_{i}', heat_map, batch_idx)

    def generate_heat_map(self,feature_map,image,classes):
        feature_map=feature_map.unsqueeze(0)
        # print(image.shape) # torch.Size([3, 512, 512])

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
                                (image.shape[1],image.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
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
        # normalize the image
        # img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        # print min and max value of the img_np
        # print("image ",np.min(img_np),np.max(img_np))
        # print(classes)
        for k in range(len(classes)):
            cam_ = cam[k].squeeze().cpu().data.numpy()
        
            # displaied as infera red image 
            # cam_pil = Image.fromarray(np.uint8(matplotlib.cm.hot(cam_)*255)).convert("RGB")
        
            # displaied black and red for important area
            # cam_pil = Image.fromarray(np.uint8(matplotlib.cm.afmhot(cam_))*100).convert("RGB")
        
            # colored but slightly blue
            cam_pil = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(cam_)*255)).convert("RGB")
            
            # print min and max value of the cam_pil
            # print(np.min(cam_pil),np.max(cam_pil))

            heatmapList.append(np.array(cam_pil) )
            
        # Calculate the blended image
        alpha = 0.8 # Alpha value for blending (adjust as needed)
        # blended_image_np = img_np.copy().astype(np.float32)
        # add all images in the heatmapList in one new image 
        blended_image_np = np.zeros_like(img_np).astype(np.float32)
        for cam_np in heatmapList:
            blended_image_np += alpha * cam_np
        # for cam_np in heatmapList:
        #     blended_image_np += alpha * cam_np
        blended_image_np/=13
        blended_image_np += img_np.astype(np.float32)
        # Clip the pixel values to the valid range [0, 255]
        blended_image_np = np.clip(blended_image_np, 0, 255).astype(np.uint8)
        # Convert the resulting array back to a PIL image
        blended_image_pil = Image.fromarray(blended_image_np).convert("RGB")
        # Display or save the blended image in rgb formate
        # blended_image_pil.show()
        # plot img_np
        # plt.imshow(img_np)
        # print(classes)
        # plt.show()
        #convert to tensor
        blended_image_pil = torchvision.transforms.functional.to_tensor(blended_image_pil)
        return blended_image_pil    

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
  

def init_working_space():

    # Creating tensorboard folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_board_folder_path="./tensor_boards/" + "heat_maps/" + str(RUN)+ f"/eval_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")

    return tensor_board_folder_path

def main():
    
    logging.info(" X_Reporto Evaluation Started")
    print(" X_Reporto Evaluation Started")
    # Logging Configurations
    log_config()
    if OperationMode.EVALUATION.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Evaluation Mode")
    
    # Tensor Board
    tensor_board_folder_path=init_working_space()
    tensor_board_writer=SummaryWriter(tensor_board_folder_path)

    # X-Reporto Trainer Object
    heat_map_model = HeatMap()

    # Create an XReportoTrainer instance with the X-Reporto model
    evaluator = HeatMapEvaluation(model=heat_map_model,tensor_board_writer=tensor_board_writer)

    # Start Training
    evaluator.evaluate()
        

if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/heat_map_Evaluator.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
    

     
# python -m src.heat_map.evaluation.heat_map_evaluation