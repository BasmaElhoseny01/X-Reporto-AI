# Logging
import matplotlib
from matplotlib import pyplot as plt
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
import torchvision

from PIL import Image

# Modules
from src.heat_map_U_ones.models.heat_map import HeatMap
from src.heat_map_U_ones.data_loader.dataset import HeatMapDataset
from src.utils import load_model

# Utils 
from config import *

class HeatMapGeneration():
        def __init__(self, model:HeatMap,evaluation_csv_path:str = heat_map_evaluation_csv_path,tensor_board_writer:SummaryWriter=None):
            self.tensor_board_writer=tensor_board_writer
            
            # Model
            self.model=model
            self.model.to(DEVICE)
            # Model in evlaution Mode Only
            self.model.eval()
            
            #---- Initialize the weights
            #print(list(self.model.model.features.parameters())[-2].shape) # 1024 Features for the dense net
            self.weights = list(self.model.model.features.parameters())[-2]
                                
                                
            # Data
            self.dataset_eval = HeatMapDataset(dataset_path= evaluation_csv_path, transform_type='test')
            logging.info(f"Evaluation dataset loaded Size: {len(self.dataset_eval)}")   

            self.data_loader_eval = DataLoader(dataset=self.dataset_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)        
            logging.info(f"Evaluation DataLoader Loaded Size: {len(self.data_loader_eval)}")
            
            
            
        def generate(self):
            with torch.no_grad():
                logging.info(f"Generating Heat Map Every{GENERATE_HEAT_MAP_EVERY}")
                for batch_idx,(images,targets,images_path) in enumerate(self.data_loader_eval):
                    
                    # Move inputs to Device
                    images = images.to(DEVICE)
                    targets=targets.to(DEVICE)
                    
                    # Forward Pass
                    _,_,features=self.model(images)#torch.Size([2, 1024, 7, 7])                    
                    
                    # For Each Image in the Batch Generate the heat Map
                    for i , img in enumerate(images):
                        self.generate_one_heat_map(images_path[i],features[i])
                        
                return 

                
        def generate_one_heat_map(self,image_path,features):
            #---- Generate heatmap
            print(features.shape) #torch.Size([1024, 7, 7])
                        
            heatmap = None
            for i in range (0, len(self.weights)):
                map = features[i,:,:]
                if i == 0: heatmap = self.weights[i] * map
                else: heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy() #torch.Size([7, 7])
            
            
            
            #---- Blend original and heatmap 
            imgOriginal = cv2.imread(image_path, 1)
            imgOriginal = cv2.resize(imgOriginal, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE)) #(224, 224, 3)

            cam = npHeatmap / np.max(npHeatmap)
            cam = cv2.resize(cam, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE))
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            
            img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Heat Map Name
            # Split the file path by the directory separator '/'
            parts = image_path.split('/')
        

            # Get the last two directory names and the file name
            last_two_directories = '_'.join(parts[-3:-1])
            file_name = parts[-1]

            # Construct the desired string
            desired_string = f"{last_two_directories}_{file_name}"
            
            plt.imshow(img)
            plt.plot()
            plt.axis('off')
            plt.savefig(f"./tensor_boards/{RUN}/{desired_string}")
            plt.show()
            return
        

def init_working_space():

    # Creating tensorboard folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_datetime="test"
    tensor_board_folder_path="./tensor_boards/" + "heat_maps/" + str(RUN)+ f"/generation_{current_datetime}"
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
    setup_logging(log_file_path='./logs/generate_heat_map.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)

# python -m src.heat_map_U_ones.evaluation.generate_heat_map