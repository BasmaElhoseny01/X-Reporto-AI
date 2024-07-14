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
from src.utils import save_model,load_model,save_checkpoint,load_checkpoint,seed_worker,ROC_AUC,plot_to_image
from config import *


import matplotlib.pyplot as plt
import numpy as np

class HeatMapTrainer:
    """
    Class to train a heat map model using PyTorch and manage training and validation processes.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        tensor_board_writer (SummaryWriter): TensorBoard SummaryWriter for logging.
        training_csv_path (str): Path to the training CSV file.
        validation_csv_path (str): Path to the validation CSV file.
    """

    def __init__(self, model:None,tensor_board_writer:SummaryWriter,training_csv_path: str =heat_map_training_csv_path,validation_csv_path:str = heat_map_validating_csv_path):
        """
        Initialize the HeatMapTrainer instance.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            tensor_board_writer (SummaryWriter): TensorBoard SummaryWriter for logging.
            training_csv_path (str, optional): Path to the training CSV file. Defaults to heat_map_training_csv_path.
            validation_csv_path (str, optional): Path to the validation CSV file. Defaults to heat_map_validating_csv_path.
        """
        self.tensor_board_writer=tensor_board_writer
        
        self.model = model
        # Continue Training
        if CONTINUE_TRAIN:
            logging.info("Loading heat_map ....")
            load_model(model=self.model,name='heat_map_best')    
            
        # Move to device
        self.model.to(DEVICE)

        # Adam Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, betas=(LR_BETA_1, LR_BETA_2))
        
        # Create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)
        
        #Create Criterion
        self.criterion=nn.BCEWithLogitsLoss(reduction='sum')

        
        # create dataset
        self.dataset_train = HeatMapDataset(dataset_path= training_csv_path, transform_type='train')
        logging.info(f"Train dataset loaded Size: {len(self.dataset_train)}")   
        
        self.dataset_val = HeatMapDataset(dataset_path= validation_csv_path, transform_type='val')
        logging.info(f"Validation dataset loaded Size: {len(self.dataset_val)}")

        #create data loader
        g = torch.Generator()
        g.manual_seed(SEED)
        # [Fix] No of Workers & Shuffle
        self.data_loader_train = DataLoader(dataset=self.dataset_train,batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                            worker_init_fn=seed_worker, generator=g)
        logging.info(f"Training DataLoader Loaded Size: {len(self.data_loader_train)}")
        
        self.data_loader_val = DataLoader(dataset=self.dataset_val,batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        logging.info(f"Validation DataLoader Loaded Size: {len(self.data_loader_val)}")
        
        # Best Loss
        self.best_loss = float('inf')

    
    def train(self,start_epoch=0,epoch_loss_init=0,start_batch=0):
        """
        Train the model over multiple epochs.

        Args:
            start_epoch (int, optional): Starting epoch for training. Defaults to 0.
            epoch_loss_init (int, optional): Initial epoch loss. Defaults to 0.
            start_batch (int, optional): Starting batch index. Defaults to 0.
        """
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
                    # raise Exception("CRASH")         
 
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)

                # Forward Pass
                total_loss=self.forward_pass(epoch,batch_idx,images,targets)
                # print("total_loss",total_loss)

                epoch_loss+=total_loss.item()              
     
                # backward Pass
                total_loss.backward()
                
                # Accumulation Learning
                if (batch_idx+1) % ACCUMULATION_STEPS==0:
                    # update the parameters
                    self.optimizer.step()
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # logging.debug(f'[Accumulative Learning after {batch_idx+1} steps ] Update Weights at  epoch: {epoch+1},'+f'Batch {batch_idx + 1}/{len(self.data_loader_train)} ')
                    del images
                    del targets

                torch.cuda.empty_cache()
                gc.collect()  

                if (batch_idx+1)%AVERAGE_EPOCH_LOSS_EVERY==0:
                    # Every 100 Batch print Average Loss for epoch till Now
                    logging.info(f'[Every {AVERAGE_EPOCH_LOSS_EVERY} Steps]: Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_train)},'+
                                 f'Average Epoch Loss : {epoch_loss/(batch_idx+1):.4f}')

                    # [Tensor Board]: Epoch Average loss
                    self.tensor_board_writer.add_scalar(f'Training/Epoch Average Loss [Every {AVERAGE_EPOCH_LOSS_EVERY} Steps]'
                                                      ,epoch_loss/(batch_idx+1)
                                                      ,epoch * len(self.data_loader_train) + batch_idx)
                                 
                # Checkpoint every N steps inside epochs        
                total_steps+=1            
                if(total_steps%CHECKPOINT_EVERY_N == 0):
                    save_checkpoint(epoch=epoch,batch_index=batch_idx,optimizer_state=self.optimizer.state_dict(),
                                    scheduler_state_dict=self.lr_scheduler.state_dict(),model_state=self.model.state_dict(),
                                    best_loss=self.best_loss,epoch_loss=epoch_loss)
                    total_steps=0
            # END OF EPOCH
                
            # [Logging]: Average Loss for epoch where each image is seen once
            logging.info(f'Epoch {epoch+1}/{EPOCHS}, Average epoch loss : {epoch_loss/(len(self.data_loader_train)):.4f}')
            # [Tensor Board]: Epoch Average loss
            self.tensor_board_writer.add_scalar('Training/Epoch Average Loss',epoch_loss/(len(self.data_loader_train)),epoch+1)
            
            # Free GPU memory
            del total_loss
            torch.cuda.empty_cache()
            gc.collect()


            # [TODO] Reove this just for prevent crash
            self.save_model(model=self.model,name="heat_map",epoch=epoch,validation_loss=5)

            # Compute ROC
            # [TODO] Fix
            self.model.optimal_thresholds=self.compute_training_ROC(epoch=epoch)
            
            
            # validate the model no touch :)
            self.model.eval()
            optimal_thresholds,validation_average_loss= self.validate_during_training(epoch=epoch) 
            logging.info(f'Validation Average Loss: {validation_average_loss:.4f}')
            self.tensor_board_writer.add_scalar('Validation/Loss',validation_average_loss,epoch+1)
            # Optimal Thresholds
            self.model.train()     

            self.lr_scheduler.step()
            # Get the new learning rate
            new_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{EPOCHS},Learning Rate: {new_lr:.10f}")
            
            # [Tensor Board]: Learning Rate
            self.tensor_board_writer.add_scalar('Training/Learning Rate',new_lr,epoch+1)


            # Add Optimal Thresholds to the Model
            # self.model.optimal_thresholds=optimal_thresholds

            # saving model per epoch
            self.save_model(model=self.model,name="heat_map",epoch=epoch,validation_loss=validation_average_loss)


        logging.info("Training Done")
        
    def compute_training_ROC(self,epoch):
        """
        Compute ROC thresholds for training data.

        Args:
            epoch (int): Current epoch number.

        Returns:
            list: List of optimal thresholds for each class.
        """

        # Option(1) Use 0.5 as threshold for all classes for faster computation in training
        logging.info("Computing ROC for Training [0.5]*10.....")
        optimal_thresholds=0.5*np.ones(len(CLASSES))  # [STOP]    
        return optimal_thresholds
    
        # Option(2) Compute ROC for Training
        all_preds= np.zeros((1, len(CLASSES)))
        all_targets= np.zeros((1, len(CLASSES)))
            
        with torch.no_grad():
            for batch_idx, (images, targets,_) in enumerate(self.data_loader_val):
                if batch_idx % 10 == 0:
                    logging.info(f"[Every 10 batches]Computing ROC for Training Batch {batch_idx + 1}/{len(self.data_loader_train)}")
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)

                # Forward Pass
                _,y_scores,_=self.model(images)  # Return is y_pred,y_scores

                mask = (targets != -1).float()

                # apply mask to the targets
                targets = targets * mask

                # y_pred = y_pred * mask
                
                # Cumulate all predictions ans labels
                all_preds = np.concatenate((all_preds, y_scores.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)
                all_targets = np.concatenate((all_targets, targets.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)

                del images
                del targets
                del mask
                torch.cuda.empty_cache()
                gc.collect()

        optimal_thresholds=self.Validation_ROC_AUC(epoch=epoch,y_true=all_targets[1:,:],y_scores=all_preds[1:,:],tensor_board_card="Training")
        return optimal_thresholds
                

    def forward_pass(self,epoch:int,batch_idx:int,images:torch.Tensor,targets:torch.Tensor,validate_during_training=False):
        """
        Perform a forward pass through the model and calculate loss.

        Args:
            epoch (int): Current epoch number.
            batch_idx (int): Current batch index.
            images (torch.Tensor): Input images.
            targets (torch.Tensor): Target labels.
            validate_during_training (bool, optional): Whether validating during training. Defaults to False.

        Returns:
            torch.Tensor: Total loss.
        """
        # Forward Pass
        y_pred,y_scores,_=self.model(images)  # Return is y_pred,y_scores

        # targets shape is [batch_size, num_labels]
        # y_pred shape is [batch_size, num_labels]
        # create mask for the targets if the target is -1 then it is 0 else 1
        mask = (targets != -1).float() # NOT useful in case of nan --> -1

        # apply mask to the targets
        targets = targets * mask
        y_pred = y_pred * mask


        # VIP DON'T FORGET TO UPDATE ONE IN EVALUATION :D
        Total_loss=self.criterion(y_pred,targets)/BATCH_SIZE

        if(Total_loss>100000):
            Total_loss=0.0

        # del 
        del y_pred
        torch.cuda.empty_cache()
        gc.collect()

        if not validate_during_training:
            logging.debug(f"epoch: {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_train)} heatmap_Loss: {Total_loss:.4f}")
            # [Tensor Board]: Avg Batch Loss
            self.tensor_board_writer.add_scalar('Training/Avg Batch Losses',Total_loss,epoch * len(self.data_loader_train) + batch_idx)

            return Total_loss
        else:
            logging.debug(f'Validation epoch: {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_val)} heatmap_Loss: {Total_loss:.4f}')
            # [Tensor Board]: Avg Batch Loss 
            self.tensor_board_writer.add_scalar('Validation/Avg Batch Losses',Total_loss,epoch * len(self.data_loader_train) + batch_idx)

            return y_scores,Total_loss
    

    
    def validate_during_training(self,epoch):
        """
        Validate the model on the validation set during training.

        Args:
            epoch (int): Current epoch number.

        Returns:
            tuple: Tuple containing optimal thresholds and validation average loss.
        """
        
        all_preds= np.zeros((1, len(CLASSES)))
        all_targets= np.zeros((1, len(CLASSES)))

        with torch.no_grad():
            validation_total_loss=0
            total_loss=0
            for batch_idx, (images, targets,_) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE)
                
                # Forward Pass
                y_scores,total_loss=self.forward_pass(epoch=epoch,batch_idx=batch_idx,images=images,targets=targets,validate_during_training=True)
                validation_total_loss+=total_loss

                # Cumulate all predictions ans labels
                # all_preds = np.concatenate((all_preds, y_scores.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)
                # all_targets = np.concatenate((all_targets, targets.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)


                del images
                del targets
                del y_scores
                torch.cuda.empty_cache()
                gc.collect() 

            # average validation_total_loss
            validation_total_loss/=(len(self.data_loader_val))

            # update the learning rate according to the validation loss if decrease
            # self.lr_scheduler.step(validation_total_loss)
    
            # Compute ROC_AUC
            # optimal_thresholds=self.Validation_ROC_AUC(epoch=epoch,y_true=all_targets[1:,:],y_scores=all_preds[1:,:])            
            optimal_thresholds=0.5*np.ones(len(CLASSES))  # [STOP]    

            return optimal_thresholds,validation_total_loss

    def Validation_ROC_AUC(self,epoch,y_true,y_scores,tensor_board_card="Validation"):
        """
        Compute ROC AUC for validation.

        Args:
            epoch (int): Current epoch number.
            y_true (np.array): True labels.
            y_scores (np.array): Predicted scores.
            tensor_board_card (str): TensorBoard card name.

        Returns:
            list: List of optimal thresholds for each class.
        """

        plt.figure(figsize=(10, 8))  # Adjust figure size

        optimal_thresholds=[]

        # Draw ROC Curve for Each Class
        for i in range(len(CLASSES)):    
            x = y_true[:, i]
            x[x<0] = 0
            x[x>1] =1
            y_true[:, i] = x 
            fpr, tpr,auc,optimal_threshold = ROC_AUC(y_true[:, i], y_scores[:, i])

            optimal_thresholds.append(optimal_threshold)

            # Plotting
            # Plot Line with optimal threshold in legend
            plt.plot(fpr, tpr, label=CLASSES[i] + ' (AUC = %0.2f, Optimal Threshold = %0.2f)' % (auc, optimal_threshold), linewidth=2)
      
        # Add legend, labels, and grid
        plt.legend(loc='lower right', fontsize=8)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.title('ROC Curves for Different Classes', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)  # Decrease tick label font size
        plt.grid(False)

        # Convert the plot to a tensor
        image = plot_to_image()

        # Write the image to the event file
        self.tensor_board_writer.add_image(f'{tensor_board_card}/ROC_curve', image, global_step=epoch,dataformats='HWC')
        logging.info("ROC Added To Tensor board"+f'{tensor_board_card}/ROC_curve')

        # [Tensor Board] to the event file
        for idx, threshold in enumerate(optimal_thresholds):
          self.tensor_board_writer.add_scalar(f'{tensor_board_card}/Optimal_Threshold_{CLASSES[idx]}', threshold, global_step=epoch)

        return optimal_thresholds
        
    def save_model(self,model:torch.nn.Module,name:str,epoch:int,validation_loss:float):
        """
        Save the model's state and optimizer's state.

        Args:
            model (torch.nn.Module): Model to be saved.
            name (str): Name of the model.
            epoch (int): Current epoch number.
            validation_loss (float, optional): Validation loss. Defaults to 0.0.
        """
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
    # current_datetime="test"
    tensor_board_folder_path="./tensor_boards/" +  str(RUN) + f"/train_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")

    return models_folder_path,ck_folder_path,tensor_board_folder_path

def main():
    """
    Main training function to start the training process.
    """
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