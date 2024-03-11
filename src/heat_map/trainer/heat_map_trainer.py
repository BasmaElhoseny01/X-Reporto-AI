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
from src.heat_map.models.heat_map import HeatMap
from src.heat_map.data_loader.dataset import HeatMapDataset

Utils
from src.utils import save_model,save_checkpoint,load_checkpoint,seed_worker
from config import *


class HeatMapTrainer:
    def __init__(self, model:None,tensor_board_writer:SummaryWriter,training_csv_path: str =heat_map_training_csv_path,validation_csv_path:str = heat_map_validating_csv_path):
        self.model = model

        self.tensor_board_writer=tensor_board_writer

        # Move to device
        self.model.to(DEVICE)

        # create adam optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr= LEARNING_RATE, weight_decay=0.0005)

        # Create Criterion 
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss  -(y log(p)+(1-y)log(1-p))    

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=SCHEDULAR_GAMMA, patience=SCHEDULAR_STEP_SIZE, threshold=THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)

        # create dataset
        self.dataset_train = HeatMapDataset(dataset_path= training_csv_path, transform_type='train')
        logging.info("Train dataset loaded")        
        self.dataset_val = HeatMapDataset(dataset_path= validation_csv_path, transform_type='val')
        logging.info("Validation dataset loaded")

        # create data loader
        g = torch.Generator()
        g.manual_seed(SEED)
        self.data_loader_train = DataLoader(dataset=self.dataset_train,batch_size=BATCH_SIZE, shuffle=True, num_workers=8, worker_init_fn=seed_worker, generator=g)
        logging.info(f"Training DataLoader Loaded Size: {len(self.data_loader_train)}")
        self.data_loader_val = DataLoader(dataset=self.dataset_val,batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        logging.info(f"Validation DataLoader Loaded Size: {len(self.data_loader_val)}")


    def train(self,start_epoch=0,epoch_loss_init=0,start_batch=0):
        # make model in training mode
        logging.info("Start Training")
        self.model.train()


        total_steps=0
        for epoch in range(start_epoch, start_epoch + EPOCHS):
            if epoch==start_epoch:
                # Loaded loss from chkpt
                epoch_loss=epoch_loss_init
            else:
                epoch_loss = 0
            for batch_idx, (images, targets) in enumerate(self.data_loader_train):
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
                # convert targets to float
                targets=targets.type(torch.float32)

                # Forward Pass
                total_loss=self.forward_pass(epoch,batch_idx,images,targets)
                epoch_loss+=total_loss

                # backward pass
                total_loss.backward()


                # Accumulation Learning
                if (batch_idx+1) %ACCUMULATION_STEPS==0:
                    # update the parameters
                    self.optimizer.step()
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    logging.debug(f'[Accumulative Learning after {batch_idx+1} steps ] Update Weights at  epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} ')

                # Get the new learning rate
                new_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_train)}, Learning Rate: {new_lr:.10f}")
                # [Tensor Board]: Learning Rate
                self.tensor_board_writer.add_scalar('Learning Rate',new_lr,epoch * len(self.data_loader_train) + batch_idx)


                if (batch_idx+1)%100==0:
                    # Every 100 Batch print Average Loss for epoch till Now
                    logging.info(f'[Every 100 Batch]: Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_train)}, Average Cumulative Epoch Loss : {epoch_loss/(batch_idx+1):.4f}')
                   
                    # [Tensor Board]: Epoch Average loss
                    self.tensor_board_writer.add_scalar('Epoch Average Loss/Every 100 Step',epoch_loss/(batch_idx+1),epoch * len(self.data_loader_train) + batch_idx)
                
                # Checkpoint every N steps inside epochs
                total_steps+=1            
                if(total_steps%CHECKPOINT_EVERY_N == 0):
                    save_checkpoint(epoch=epoch,batch_index=batch_idx,optimizer_state=self.optimizer.state_dict(),
                                    scheduler_state_dict=self.lr_scheduler.state_dict(),model_state=self.model.state_dict(),
                                    best_loss=self.best_loss,epoch_loss=epoch_loss)
                    total_steps=0
                
                   
            # [Logging]: Average Loss for epoch where each image is seen once
            logging.info(f'Epoch {epoch+1}/{EPOCHS}, Average epoch loss : {epoch_loss/(len(self.data_loader_train)):.4f}')
            # [Tensor Board]: Epoch Average loss
            self.tensor_board_writer.add_scalar('Epoch Average Loss/Every Epoch',epoch_loss/(len(self.data_loader_train)),epoch+1)

                   
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

        # save the best model            
        logging.info("Training Done")
        
    def validate_during_training(self,epoch):
        '''
        Validate the model during training
        '''
        with torch.no_grad():
            validation_total_loss=0
            total_loss=0
            for batch_idx, (images, targets) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)
                # convert targets to float
                targets=targets.type(torch.float32)

                # Forward Pass
                total_loss=self.forward_pass(epoch=epoch,batch_idx=batch_idx,images=images,targets=targets,validate_during_training=True)
                validation_total_loss+=total_loss
           
            # average validation_total_loss
            validation_total_loss/=(len(self.data_loader_val))

            # update the learning rate according to the validation loss if decrease
            self.lr_scheduler.step(validation_total_loss)

            return validation_total_loss

        
    def forward_pass(self,epoch:int,batch_idx:int,images:torch.Tensor,targets:torch.Tensor,validate_during_training=False):
        # Forward Pass
        _,y=self.model(images)

        # Compute Losses
        Total_loss= 0
        # Loss as summation of Binary cross entropy for each class :D Only 13 Classes bec we removed the class of no-finding 
        for c in range(13):
            Total_loss+=self.criterion(y[:,c],targets[:,c])

        if not validate_during_training:
            logging.debug(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} heatmap_Loss: {Total_loss:.4f}')
         
            # [Tensor Board]: Avg Batch Loss
            self.tensor_board_writer.add_scalar('Avg Batch Losses',Total_loss,epoch * len(self.data_loader_train) + batch_idx)

        else:
            logging.debug(f'Validation epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_val)} object_detector_Loss: {Total_loss:.4f}')
            # [Tensor Board]: Avg Batch Loss 
            self.tensor_board_writer.add_scalar('Avg Batch Losses[Validation]',Total_loss,epoch * len(self.data_loader_train) + batch_idx)

        return Total_loss
    
    
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
    ck_folder_path="check_points/" + "heat_maps/" + str(RUN)
    if not os.path.exists(ck_folder_path):
        os.makedirs(ck_folder_path)
        logging.info(f"Folder '{ck_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{ck_folder_path}' already exists.")

    # Creating tensor_board folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_board_folder_path="./tensor_boards/" + "heat_map/" + str(RUN) + f"/train_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")

    return models_folder_path,ck_folder_path,tensor_board_folder_path

      

def main():
    logging.info("Training Heat Map Started")
    sys.exit()
    # Logging Configurations
    log_config()

    if OperationMode.TRAINING.value!=OPERATION_MODE :
            #throw exception 
            raise Exception("Operation Mode is not Training Mode")

    _,_,tensor_board_folder_path=init_working_space()
    
    # HeatMap Trainer Object
    heat_map_model = HeatMap()


    #  # Tensor Board
    tensor_board_writer=SummaryWriter(tensor_board_folder_path)

    # Create an HeatMapTrainer instance with the HeatMap model
    trainer = HeatMapTrainer(model=heat_map_model,tensor_board_writer=tensor_board_writer)
    sys.exit()

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
    setup_logging(log_file_path='./logs/heat_map_trainer.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)

# python -m src.heat_map.trainer.heat_map_trainer