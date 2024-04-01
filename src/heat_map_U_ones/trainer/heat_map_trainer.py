# Logging
from logger_setup import setup_logging
import logging

from datetime import datetime

import os
import gc
from tqdm import tqdm
import sys


from torch.utils.tensorboard import SummaryWriter

# Torch
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import  DataLoader

# Modules
from src.heat_map_U_ones.models.heat_map import HeatMap
from src.heat_map_U_ones.data_loader.dataset import HeatMapDataset

# Utils
from src.utils import save_model,load_model,save_checkpoint,load_checkpoint,seed_worker
from config import *

class HeatMapTrainer:
    def __init__(self, model:None,tensor_board_writer:SummaryWriter,training_csv_path: str =heat_map_training_csv_path,validation_csv_path:str = heat_map_validating_csv_path):
        #self.model = model
        #if CONTINUE_TRAIN:
            #logging.info("Loading heat_map ....")
            #load_model(model=self.model,name='heat_map_best')
            
        #self.tensor_board_writer=tensor_board_writer

        ## Move to device
        #self.model.to(DEVICE)

        #self.optimizer = optim
        
        #Create Criterion
        #self.criterion = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos).to(DEVICE)
 

        # Create learning rate scheduler
        #self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.5)

        # create dataset
        self.dataset_train = HeatMapDataset(dataset_path= training_csv_path, transform_type='train')
        logging.info(f"Train dataset loaded Size: {len(self.dataset_train)}")        
        #print("Train dataset loaded")    
        #img,labels=(self.dataset_train[76])   
        
        #self.dataset_val = HeatMapDataset(dataset_path= validation_csv_path, transform_type='val')
        #logging.info("Validation dataset loaded")
        #print("Validation dataset loaded")

        #create data loader
        g = torch.Generator()
        g.manual_seed(SEED)
        # [Fix] No of Workers & Shuffle
        self.data_loader_train = DataLoader(dataset=self.dataset_train,batch_size=BATCH_SIZE, shuffle=False, num_workers=1, worker_init_fn=seed_worker, generator=g)
        logging.info(f"Training DataLoader Loaded Size: {len(self.data_loader_train)}")
        
        #self.data_loader_val = DataLoader(dataset=self.dataset_val,batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        #logging.info(f"Validation DataLoader Loaded Size: {len(self.data_loader_val)}")
        #print(f"Validation DataLoader Loaded Size: {len(self.data_loader_val)}")
        
        # Best Loss
        #self.best_loss=78.1398
        #self.best_loss=10000000000.0
    
        

def init_working_space():
    # Creating run folder
    models_folder_path="models/" + str(RUN)
    if not os.path.exists(models_folder_path):
        os.makedirs(models_folder_path)
        logging.info(f"Folder '{models_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{models_folder_path}' already exists.")

    # Creating checkpoints folder
    ck_folder_path="check_points/" +  str(RUN)
    if not os.path.exists(ck_folder_path):
        os.makedirs(ck_folder_path)
        logging.info(f"Folder '{ck_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{ck_folder_path}' already exists.")

    # Creating tensor_board folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_board_folder_path="./tensor_boards/" +  str(RUN) + f"/train_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")

    return models_folder_path,ck_folder_path,tensor_board_folder_path

      

def main():
    logging.info("Training Heat Map Started")
    # Logging Configurations
    log_config()

    if OperationMode.TRAINING.value!=OPERATION_MODE :
            #throw exception 
            raise Exception("Operation Mode is not Training Mode")

#     _,_,tensor_board_folder_path=init_working_space()
    
#     # HeatMap Trainer Object
#     heat_map_model = HeatMap()


#     #  # Tensor Board
#     tensor_board_writer=SummaryWriter(tensor_board_folder_path)

#     # Create an HeatMapTrainer instance with the HeatMap model
    trainer = HeatMapTrainer(model=None,tensor_board_writer=None)

#     if RECOVER:
#         # Load the state of model
#         checkpoint=load_checkpoint(run=RUN)

#         # Load Model state
#         heat_map_model.load_state_dict(checkpoint['model_state'])

#         # Batch to start from
#         start_batch=checkpoint['batch_index']+1

#         # Load scheduler_state_dict
#         trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#         # Load optimizer_state
#         trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])

#         # Load best_loss
#         trainer.best_loss=checkpoint['best_loss']

#         # trainer.test_data_loader()
#         # Start Train form checkpoint ends
#         trainer.train(start_epoch=checkpoint['epoch'],epoch_loss_init=checkpoint['epoch_loss'].item(),start_batch=start_batch)

#     else:
#         # No check point
#         # Start New Training
#         trainer.train()


if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/heat_map_u_onestrainer.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)

# python -m src.heat_map_U_ones.trainer.heat_map_trainer