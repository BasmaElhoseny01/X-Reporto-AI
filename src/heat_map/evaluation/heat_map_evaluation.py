# Logging
import numpy as np
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

from torch.utils.tensorboard import SummaryWriter

# Torch
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import  DataLoader

# Modules
from src.heat_map.models.heat_map import HeatMap
from src.heat_map.data_loader.dataset import HeatMapDataset

# Utils 
from config import *
from src.utils import plot_heatmap

class HeatMapEvaluation():
    def __init__(self, model:HeatMap,evaluation_csv_path:str = evaluation_csv_path,tensor_board_writer:SummaryWriter=None):
        '''
        X-Reporto Validation Class
        Args:
        model: X-Reporto Model
        evaluation_csv_path: Path to the validation csv file
        ''' 
        self.model = model
        self.evaluation_csv_path = evaluation_csv_path
        self.tensor_board_writer=tensor_board_writer

        self.model.to(DEVICE)
        
        self.data_loader_val = DataLoader(dataset=HeatMapDataset(self.evaluation_csv_path), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        logging.info("Evaluation dataset loaded")

    def evaluate(self):
        #Evaluate the model
        scores = self.evaluate_heat_map()

        # [Tensor Board] Update the Board by the scalers for that Run
        self.update_tensor_board_score()
           

    def evaluate_heat_map(self):
        self.model.eval()
        with torch.no_grad():
            # validate the model
            logging.info("Evaluating the model")
            validation_total_loss=0

            for batch_idx,(images,targets) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)
                # convert targets to float
                targets=targets.type(torch.float32)

                # Forward Pass [TODO]
                Total_loss=self.forward_pass()

                # [Tensor Board] Draw the HeatMap Predictions of this batch
                #TODO: uncomment
                # self.draw_tensor_board(batch_idx,images)
                validation_total_loss+=Total_loss
        
        # average validation_total_loss
        validation_total_loss/=(len(self.data_loader_val))
    

    def forward_pass(self):
        Total_loss=0
        return Total_loss
    
    ########################################################### General Fuunctions ##########################################
    def update_tensor_board_score():
        pass

    def draw_tensor_board(self,batch_idx,images):
        '''
        Add images to tensorboard
        '''
        pass
        plot_heatmap()
    
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