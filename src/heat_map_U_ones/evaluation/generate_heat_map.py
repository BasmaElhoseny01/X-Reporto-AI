# Logging
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import cv2
from sklearn import metrics
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
import albumentations as A


# from PIL import Image

# Modules
from src.heat_map_U_ones.models.heat_map import HeatMap
from src.heat_map_U_ones.data_loader.dataset import HeatMapDataset
from src.utils import load_model,plot_to_image

# Utils 
from config import *


resize_and_pad_transform = A.Compose([
            A.LongestMaxSize(max_size=HEAT_MAP_IMAGE_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=HEAT_MAP_IMAGE_SIZE, min_width=HEAT_MAP_IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
        ])

class HeatMapGeneration():
        def __init__(self, model:HeatMap,evaluation_csv_path:str = heat_map_evaluation_csv_path,tensor_board_writer:SummaryWriter=None):
            self.tensor_board_writer=tensor_board_writer
            
            # Model
            self.model=model
            self.model.to(DEVICE)
            # Model in evaluation Mode Only
            self.model.eval()
            
            #---- Initialize the weights
            #print(list(self.model.model.features.parameters())[-2].shape) # 1024 Features for the dense net
            # self.weights = list(self.model.model.features.parameters())[-2]
            # get weights of prediction layer
            self.weights = list(self.model.model.classifier.parameters())[-2] #torch.Size([8, 1024])
                               
                                
            # DataSet
            self.dataset_eval = HeatMapDataset(dataset_path= evaluation_csv_path, transform_type='test')
            print(f"Evaluation dataset loaded Size: {len(self.dataset_eval)}")   

            # DataLoader
            self.data_loader_eval = DataLoader(dataset=self.dataset_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)        
            print(f"Evaluation DataLoader Loaded Size: {len(self.data_loader_eval)}")
            
            
            
        def generate(self):

            # Option(1) Explicit Set the optimal threshold from the training (Computed from evaluation on the training data :D)
            # self.model.optimal_thresholds=[ 0.2785722315311432,0.24235820770263672, 0.2138664722442627,
            # 0.2095205932855606,0.35408109426498413,0.22779232263565063,0.13535451889038086,0.3065055310726166]


            # Option(3) Use the optimal thresholds Saved in the model :D [Default]

            
            with torch.no_grad():
                logging.info(f"Generating Heat Maps")
                for batch_idx,(images,targets,images_path) in enumerate(self.data_loader_eval):
                    logging.info(f"Generating Heat Map for Batch {batch_idx+1}/{len(self.data_loader_eval)}")
                    
                    # Move inputs to Device
                    images = images.to(DEVICE)
                    targets=targets.to(DEVICE)
                    
                    # Forward Pass
                    _,y_scores,features=self.model(images)#torch.Size([2, 1024, 7, 7])                    
                    
                    # For Each Image in the Batch Generate the heat Map
                    for i , image in enumerate(images):
                        for j,class_finding in enumerate(CLASSES):                        
                          image_class_heat_map,image_title=self.generate_heat_map_per_class(image_path=images_path[i],features=features[i],class_index=j,class_name=class_finding,gold_label=targets[i][j],pred_label=y_scores[i][j],class_threshold=self.model.optimal_thresholds[j])

                          # [TensorBoard] Write Image to Board
                          self.tensor_board_writer.add_image(f'HeatMaps/{image_title}/{class_finding}', image_class_heat_map,dataformats='HWC')  

                          # [Save Heat Maps into separate image files]
                          if (True): 
                            # Define the path for saving the heat map image
                            image_filename = f'{image_title}_{class_finding}.png'
                            image_path = os.path.join('/content/heat_map_imgs/', image_filename)
                        
                            # Save the heat map image
                            cv2.imwrite(image_path, cv2.cvtColor(image_class_heat_map, cv2.COLOR_RGB2BGR))         
               
                logging.info("Done Generation")
                return 
            
        # def generate_one_heat_map(self,image_path,features,gold_labels,pred_labels,thresholds):
        # '''
        # Old Heat Map Generation Function Incorrect
        # '''
        #     #---- Generate heatmap
        #     #print(features.shape) #torch.Size([1024, 7, 7])
                        
        #     heatmap = None
        #     for i in range (0, len(self.weights)): # 1024
        #         map = features[i,:,:] # torch.Size([7, 7])
        #         if i == 0: heatmap = self.weights[i] * map #torch.Size([7, 7]) 
        #         else: heatmap += self.weights[i] * map
        #         npHeatmap = heatmap.cpu().data.numpy() #torch.Size([7, 7])
            
            
            
        #     #---- Blend original and heatmap 
        #     imgOriginal = cv2.imread(image_path, 1)
        #     imgOriginal = cv2.resize(imgOriginal, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE)) #(224, 224, 3)

        #     cam = npHeatmap / np.max(npHeatmap)
        #     cam = cv2.resize(cam, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE))
        #     heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            
        #     img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)            
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        #     # Construct the legend
        #     legend_text = "Results:\n"
        #     for index, (class_name, gold_label, pred_label) in enumerate(zip(CLASSES, gold_labels, pred_labels)):
        #         legend_text += f"{class_name}: *{gold_label}, {pred_label:.2f},{1*(pred_label>thresholds[index])}\n"

        #     # Heat Map Name
        #     # Split the file path by the directory separator '/'
        #     parts = image_path.split('/')

        #     # Get the last two directory names and the file name
        #     last_two_directories = '_'.join(parts[-3:-1])
        #     file_name = parts[-1]

        #     # Construct the desired string
        #     desired_string = f"{last_two_directories}_{file_name}"

        #     # Create subplot
        #     fig = plt.figure(figsize=(12, 5))
        #     gs = GridSpec(1, 3, width_ratios=[2, 2, 1], wspace=0.1)  # Set wspace to adjust space between subplots


        #     # Plot original image
        #     ax_original = fig.add_subplot(gs[0])
        #     ax_original.imshow(imgOriginal)
        #     ax_original.axis('off')

        #     # Plot heatmap
        #     ax_heatmap = fig.add_subplot(gs[1])
        #     ax_heatmap.imshow(img)
        #     ax_heatmap.axis('off')

        #     # Plot legend
        #     ax_legend = fig.add_subplot(gs[2])
        #     ax_legend.text(0, 0, legend_text, fontsize=10, color='black', bbox=dict(facecolor='lightgrey', alpha=0.5))
        #     ax_legend.axis('off')

        #     # Convert plot to image
        #     image = plot_to_image()

        #     # if SAVE_IMAGES:
        #     #   plt.savefig(f"./tensor_boards/heat_maps/{RUN}/{desired_string}")
        #     # plt.show()

        #     return image,desired_string
        
        def generate_heat_map_per_class(self,image_path,features,class_index:int,class_name:str,gold_label,pred_label,class_threshold):
            # Get the heatmap for the class

            #---- Generate heatmap
            #print(features.shape) #torch.Size([1024, 7, 7])

            heatmap = None
            weights = self.weights[class_index].view(1, -1) # 1, 1024
            
            # apply matrix multiplication to get the heatmap
            # weights: 1024, features: 1024, 7, 7 -> 1, 7, 7

            heatmap = torch.matmul(weights, features.view(features.size(0), -1)).view(features.size(1), features.size(2)) # 7, 7
            npHeatmap = heatmap.cpu().data.numpy() #torch.Size([7, 7])

            # apply relu
            npHeatmap = np.maximum(npHeatmap, 0)

            # apply normalization
            if(np.max(npHeatmap)!=0):
              npHeatmap = npHeatmap / np.max(npHeatmap)

            #---- Blend original and heatmap
            imgOriginal = cv2.imread(image_path, 1)
            # imgOriginal = image


            # imgOriginal = cv2.resize(imgOriginal, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE)) #(224, 224, 3) # Wrong Resize:(
            imgOriginal=resize_and_pad_transform(image=imgOriginal)["image"] # (224, 224, 3)

            cam = npHeatmap
            cam = cv2.resize(cam, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE))
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

            img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Construct the legend
            legend_text = f"Results:\n{class_name}"+f": *{gold_label}, {pred_label:.2f},{1*(pred_label>class_threshold)}\n"

            # Heat Map Name
            # Split the file path by the directory separator '/'
            parts = image_path.split('/')

            # Get the last two directory names and the file name
            last_two_directories = '_'.join(parts[-3:-1])
            file_name = parts[-1]

            # Construct the desired string with the class name
            desired_string = f"{last_two_directories}_{file_name}"
            
            # Create subplot
            fig = plt.figure(figsize=(12, 5))
            gs = GridSpec(1, 3, width_ratios=[2, 2, 1], wspace=0.1)  # Set wspace to adjust space between subplots

            # Plot original image
            ax_original = fig.add_subplot(gs[0])
            ax_original.imshow(imgOriginal)
            ax_original.axis('off')

            # Plot heatmap
            ax_heatmap = fig.add_subplot(gs[1])
            ax_heatmap.imshow(img)
            ax_heatmap.axis('off')

            # Plot legend
            ax_legend = fig.add_subplot(gs[2])
            ax_legend.text(0, 0, legend_text, fontsize=10, color='black', bbox=dict(facecolor='lightgrey', alpha=0.5))
            ax_legend.axis('off')

            # Convert plot to image
            image = plot_to_image()

            # if SAVE_IMAGES:
            #   plt.savefig(f"./tensor_boards/heat_maps/{RUN}/{desired_string}")
            # plt.show()

            return image,desired_string

        def get_gradients(self, module, grad_input, grad_output):
            self.gradients = grad_input

        def generate_heatmap_gradCam_per_class(self, image_path, features, predictions, class_index, class_name):
            '''
            Generate the gradCAM heatmap for the class
            '''

            pred = predictions[class_index]

            # register the hook
            self.model.model.features.register_backward_hook(self.model.activations_hook)
            # get gradient of the prediction with respect to the features
            
            self.model.zero_grad()
            pred.backward(retain_graph=True)

            # get the gradients of the features
            gradients = self.model.get_gradients() # 1, 1024, 7, 7

            # multiply the gradients with the features
            grad_cam = gradients * features

            # get the mean of the grad_cam
            grad_cam = grad_cam.mean(dim=1, keepdim=True).squeeze() # 7, 7

            # apply relu
            grad_cam = F.relu(grad_cam)

            # apply normalization
            grad_cam = grad_cam / torch.max(grad_cam)


            #---- Blend original and heatmap
            imgOriginal = cv2.imread(image_path, 1)

            # imgOriginal = cv2.resize(imgOriginal, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE)) #(224, 224, 3) # Wrong Resize:(
            imgOriginal=resize_and_pad_transform(image=imgOriginal)["image"] # (224, 224, 3)


            cam = grad_cam
            cam = cv2.resize(cam.cpu().data.numpy(), (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE))

            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

            img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Construct the legend

            legend_text = f"Results:\n{class_name}\n"

            # Heat Map Name
            # Split the file path by the directory separator '/'
            parts = image_path.split('/')
            # Get the last two directory names and the file name
            last_two_directories = '_'.join(parts[-3:-1])
            file_name = parts[-1]
            # Construct the desired string with the class name
            desired_string = f"{last_two_directories}_{file_name}_{class_name}"

            # Create subplot
            fig = plt.figure(figsize=(12, 5))
            gs = GridSpec(1, 3, width_ratios=[2, 2, 1], wspace=0.1)  # Set wspace to adjust space between subplots

            # Plot original image
            ax_original = fig.add_subplot(gs[0])
            ax_original.imshow(imgOriginal)
            ax_original.axis('off')

            # Plot heatmap
            ax_heatmap = fig.add_subplot(gs[1])
            ax_heatmap.imshow(img)
            ax_heatmap.axis('off')

            # Plot legend
            ax_legend = fig.add_subplot(gs[2])
            ax_legend.text(0, 0, legend_text, fontsize=10, color='black', bbox=dict(facecolor='lightgrey', alpha=0.5))
            ax_legend.axis('off')

            # Convert plot to image
            image = plot_to_image()

            # if SAVE_IMAGES:
            #   plt.savefig(f"./tensor_boards/heat_maps/{RUN}/{desired_string}")
            # plt.show()

            return image,desired_string

def init_working_space():

    # Creating tensorboard folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_board_folder_path="./tensor_boards/" + str(RUN)+ f"/generation_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")

    return tensor_board_folder_path

def main():
    logging.info("Heat Map Generation Started")
    
    # Logging Configurations
    log_config()

    if OperationMode.EVALUATION.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Evaluation Mode")
        
    # Tensor Board
    tensor_board_folder_path=init_working_space()
    tensor_board_writer=SummaryWriter(tensor_board_folder_path)
    
    
    # Heat Map Model 
    heat_map_model = HeatMap()

    logging.info("Loading heat_map ....")
    load_model(model=heat_map_model,name='heat_map_best')
    
    # Create an HeatMap Generator instance with the HeatMap model
    generator = HeatMapGeneration(model=heat_map_model,tensor_board_writer=tensor_board_writer)
    
    
    # Start Generation
    generator.generate()
    

 
    

if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/heat_map_u_ones_generator.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)

# python -m src.heat_map_U_ones.evaluation.generate_heat_map