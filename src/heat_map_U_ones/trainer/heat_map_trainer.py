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


import matplotlib.pyplot as plt
import numpy as np

class HeatMapTrainer:
    def __init__(self, model:None,tensor_board_writer:SummaryWriter,training_csv_path: str =heat_map_training_csv_path,validation_csv_path:str = heat_map_validating_csv_path):
        self.tensor_board_writer=tensor_board_writer
        
        self.model = model
        # Continue Training
        if CONTINUE_TRAIN:
            logging.info("Loading heat_map ....")
            load_model(model=self.model,name='heat_map_best')    
            
        # Move to device
        self.model.to(DEVICE)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, betas=(LR_BETA_1, LR_BETA_2))
        
        # Create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)
        
        #Create Criterion
        #self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(POS_WEIGHTS).to(DEVICE))
        # add one to the positive weights
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=(10* (1-torch.tensor(POS_WEIGHTS)) ).to(DEVICE))
        # self.criterion=nn.BCEWithLogitsLoss(reduction='mean')

        
        # create dataset
        self.dataset_train = HeatMapDataset(dataset_path= training_csv_path, transform_type='train')
        logging.info(f"Train dataset loaded Size: {len(self.dataset_train)}")   
        
        self.dataset_val = HeatMapDataset(dataset_path= validation_csv_path, transform_type='val')
        logging.info(f"Validation dataset loaded Size: {len(self.dataset_val)}")

        #create data loader
        g = torch.Generator()
        g.manual_seed(SEED)
        # [Fix] No of Workers & Shuffle
        self.data_loader_train = DataLoader(dataset=self.dataset_train,batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                            worker_init_fn=seed_worker, generator=g)
        logging.info(f"Training DataLoader Loaded Size: {len(self.data_loader_train)}")
        
        self.data_loader_val = DataLoader(dataset=self.dataset_val,batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        logging.info(f"Validation DataLoader Loaded Size: {len(self.data_loader_val)}")
        print(f"Validation DataLoader Loaded Size: {len(self.data_loader_val)}")
        
        # Best Loss
        self.best_loss = float('inf')
    
    
    def train(self,start_epoch=0,epoch_loss_init=0,start_batch=0):
        logging.info("Start Training")
        
        # make model in training mode
        self.model.train()

        total_steps=0
        for epoch in range(start_epoch, start_epoch + EPOCHS):
            #running_loss = 0.0
            if epoch==start_epoch:
                # Loaded loss from ckpt
                epoch_loss=epoch_loss_init
            else:
                epoch_loss = 0
            for batch_idx, (images, targets,_) in enumerate(self.data_loader_train):
                if batch_idx < start_batch:
                    continue  # Skip batches until reaching the desired starting batch number
        
                # Test Recovery
                # if epoch==3 and batch_idx==1:
                    # print("Start Next time from")
                    # print(object_detector_targets[1])
                    # print(batch_idx)
                    # raise Exception("CRASSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHHHH")         
 
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)

                # Forward Pass
                total_loss=self.forward_pass(epoch,batch_idx,images,targets)

                epoch_loss+=total_loss.item()              
     
                # backward Pass
                total_loss.backward()
                
                # Acculmulation Learning
                if (batch_idx+1) % ACCUMULATION_STEPS==0:
                    # update the parameters
                    self.optimizer.step()
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    #logging.debug(f'[Accumlative Learning after {batch_idx+1} steps ] Update Weights at  epoch: {epoch+1},'+f'Batch {batch_idx + 1}/{len(self.data_loader_train)} ')
                  
                                
        
                if (batch_idx+1)%AVERGAE_EPOCH_LOSS_EVERY==0:
                    # Every 100 Batch print Average Loss for epoch till Now
                    logging.info(f'[Every {AVERGAE_EPOCH_LOSS_EVERY} Batch]: Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_train)},'+
                                 f'Average Cumulative Epoch Loss : {epoch_loss/(batch_idx+1):.4f}')

                    # [Tensor Board]: Epoch Average loss
                    #self.tensor_board_writer.add_scalar('Epoch Average Loss/Every 100 Step'
                    #                                   ,epoch_loss/(batch_idx+1)
                    #                                   ,epoch * len(self.data_loader_train) + batch_idx)
                    
                    
            # END OF EPOCH
            # Checkpoint every N steps inside epoches        
            total_steps+=1            
            if(total_steps%CHECKPOINT_EVERY_N == 0):
                save_checkpoint(epoch=epoch,batch_index=batch_idx,optimizer_state=self.optimizer.state_dict(),
                                scheduler_state_dict=self.lr_scheduler.state_dict(),model_state=self.model.state_dict(),
                                best_loss=self.best_loss,epoch_loss=epoch_loss)
                total_steps=0
                
            # [Logging]: Average Loss for epoch where each image is seen once
            logging.info(f'Epoch {epoch+1}/{EPOCHS}, Average epoch loss : {epoch_loss/(len(self.data_loader_train)):.4f}')
            # [Tensor Board]: Epoch Average loss
            #self.tensor_board_writer.add_scalar('Epoch Average Loss/Every Epoch',epoch_loss/(len(self.data_loader_train)),epoch+1)
            
            # Free GPU memory
            del total_loss
            torch.cuda.empty_cache()
            gc.collect()
            
            
            # validate the model no touch :)
            self.model.eval()
            validation_average_loss= self.validate_during_training(epoch=epoch) 
            logging.info(f'Validation Average Loss: {validation_average_loss:.4f}')
            self.tensor_board_writer.add_scalar('Average [Validation] Loss/Every Epoch',validation_average_loss,epoch+1)
            self.model.train()     
                       
            # saving model per epoch
            self.save_model(model=self.model,name="heat_map",epoch=epoch,validation_loss=validation_average_loss)

        logging.info("Training Done")
        
            
    def forward_pass(self,epoch:int,batch_idx:int,images:torch.Tensor,targets:torch.Tensor,validate_during_training=False):
        # Forward Pass
        y_pred,_,_=self.model(images)  # Return is y_pred,y_scores
        
        # VIP DON'T FORGET TO UPDATE ONE IN EVALUATION :D
        Total_loss=self.criterion(y_pred,targets)*images[0].size(0)   #3-channels
       

        if not validate_during_training:
            # logging.debug(f"epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} heatmap_Loss: {Total_loss:.4f}")
         
            # [Tensor Board]: Avg Batch Loss
            #self.tensor_board_writer.add_scalar('Avg Batch Losses',Total_loss,epoch * len(self.data_loader_train) + batch_idx)
            pass

        else:
            # logging.debug(f'Validation epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_val)} heatmap_Loss: {Total_loss:.4f}')
            # [Tensor Board]: Avg Batch Loss 
            #self.tensor_board_writer.add_scalar('Avg Batch Losses[Validation]',Total_loss,epoch * len(self.data_loader_train) + batch_idx)
            pass
        return Total_loss
    

    
    def validate_during_training(self,epoch):
        '''
        Validate the model during training
        '''
        with torch.no_grad():
            validation_total_loss=0
            total_loss=0
            for batch_idx, (images, targets,_) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)
                
                # Forward Pass
                total_loss=self.forward_pass(epoch=epoch,batch_idx=batch_idx,images=images,targets=targets,validate_during_training=True)
                validation_total_loss+=total_loss
           
            # average validation_total_loss
            validation_total_loss/=(len(self.data_loader_val))

            # update the learning rate according to the validation loss if decrease
            self.lr_scheduler.step(validation_total_loss)

            return validation_total_loss

        
        
    def save_model(self,model:torch.nn.Module,name:str,epoch:int,validation_loss:float):
        '''
        Save the current state of model.
        '''
        logging.info("Saving "+name+"_epoch "+str(epoch+1))
        save_model(model=model,name=name+"_epoch_"+str(epoch+1))
        self.check_best_model(epoch,validation_loss,name,model)  
        
    def check_best_model(self,epoch:int,validation_loss:float,name:str,model:torch.nn.Module):
        '''
        Check if the current model is the best model
        '''
        if(validation_loss<=self.best_loss) :
                self.best_loss=validation_loss
                save_model(model=model,name=name+"_best")
                logging.info(f"Best Model Updated: {name}_best at epoch {epoch+1} with Average validation loss: {self.best_loss:.4f}")   

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
    current_datetime="test"
    tensor_board_folder_path="./tensor_boards/heat_maps/" +  str(RUN) + f"/train_{current_datetime}"
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

    _,_,tensor_board_folder_path=init_working_space()
    
    # HeatMap Trainer Object
    heat_map_model = HeatMap() 

     # Tensor Board
    tensor_board_writer=SummaryWriter(tensor_board_folder_path)

    # Create an HeatMapTrainer instance with the HeatMap model
    trainer = HeatMapTrainer(model=heat_map_model,tensor_board_writer=tensor_board_writer)

    if RECOVER:
        # Load the state of model
        checkpoint=load_checkpoint(run=RUN)

        # Load Model state
        heat_map_model.load_state_dict(checkpoint['model_state'])

        # Batch to start from
        start_batch=checkpoint['batch_index']+1

        # Load scheduler_state_dict
        trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load optimizer_state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Load best_loss
        trainer.best_loss=checkpoint['best_loss']

        # trainer.test_data_loader()
        # Start Train form checkpoint ends
        trainer.train(start_epoch=checkpoint['epoch'],epoch_loss_init=checkpoint['epoch_loss'].item(),start_batch=start_batch)

    else:
        # No check point
        # Start New Training
        trainer.train()


if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/heat_map_u_ones_trainer.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)

# python -m src.heat_map_U_ones.trainer.heat_map_trainer