# Logging
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
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
from src.utils import load_model,ROC_AUC,plot_to_image

# Utils 
from config import *

class HeatMapEvaluation():
    def __init__(self, model:HeatMap,evaluation_csv_path:str = heat_map_evaluation_csv_path,tensor_board_writer:SummaryWriter=None):
        '''
        X-Reporto Validation Class
        Args:
        model: X-Reporto Model
        evaluation_csv_path: Path to the validation csv file
        ''' 
        self.model=model
        self.model.to(DEVICE)
        
        self.tensor_board_writer=tensor_board_writer
        
        self.dataset_eval = HeatMapDataset(dataset_path= evaluation_csv_path, transform_type='test')
        logging.info(f"Evaluation dataset loaded Size: {len(self.dataset_eval)}")   

        self.data_loader_eval = DataLoader(dataset=self.dataset_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)        
        logging.info(f"Evaluation DataLoader Loaded Size: {len(self.data_loader_eval)}")
               
        
    def evaluate(self):
        #Evaluate the model
        logging.info("Evaluating the model")

        # Run Predictions
        eval_loss,all_preds,all_targets = self.evaluate_heat_map()
        logging.info(f"Evaluation Loss :{eval_loss}")     

        # Compute Metrics
        # Compute AUC
        auc= self.compute_AUC(y_true=all_targets,y_scores=all_preds)
        logging.info("AUC: %s", auc)
                    
        # Compute Fp....
        classification_metrics=self.compute_classification_metrics(y_true=all_targets,y_pred=all_preds,thresholds=self.model.optimal_thresholds)
        for metric,values in classification_metrics.items():
          logging.info(f"{metric}:{values}")

        # [Tensor Board] 
        self.save_result_to_tensor_board(evaluation_loss=eval_loss,aucs=auc,classification_metrics=classification_metrics)
        

    def evaluate_heat_map(self):
        # Init The Scores
        all_preds= np.zeros((1, len(CLASSES)))
        all_targets= np.zeros((1, len(CLASSES)))
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx,(images,targets,_) in enumerate(self.data_loader_eval):
                logging.info(f"Batch {batch_idx}/{len(self.data_loader_eval)} ....")

                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE) 

                # Forward Pass [TODO]
                _,loss,scores=self.forward_pass(images,targets)
                
                evaluation_total_loss=loss

                logging.info(f"Batch {batch_idx}/{len(self.data_loader_eval)} Loss: {evaluation_total_loss}")
                        
                # Cumulate all predictions ans labels
                all_preds = np.concatenate((all_preds, scores.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)
                all_targets = np.concatenate((all_targets, targets.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)  

                del images
                del targets

            gc.collect()
            torch.cuda.empty_cache()       
            
        return evaluation_total_loss,all_preds[1:,:],all_targets[1:,:]
    
    
    def forward_pass(self,images,targets):
        '''
        y: Prob not classes
        '''
        # Forward Pass
        y_pred,scores,_=self.model(images)
        mask = (targets != -1).float()

        # apply mask to the targets
        targets = targets * mask
        y_pred = y_pred * mask

        # Calculate Loss [CHECK the loss function as in the training script]
        Total_loss=nn.BCEWithLogitsLoss(reduction='sum')(y_pred,targets)/BATCH_SIZE
        
        features=None
        
        return features,Total_loss,scores

    def compute_AUC(self,y_true,y_scores):
        AUCs={}
        plt.figure(figsize=(10, 8))  # Adjust figure size

        for i in range(len(CLASSES)):    
            x = y_true[:, i]
            x[x<0] = 0
            x[x>1] =1
            y_true[:, i] = x 
            fpr, tpr, thresholds = metrics.roc_curve(y_true[:, i], y_scores[:, i])
            # AUC
            auc = metrics.auc(fpr, tpr)

            # Optimal Threshold
            # Compute Youden's J statistic
            j_statistic = tpr - fpr

            # Find the index of the threshold that maximizes J statistic
            optimal_threshold_index = np.argmax(j_statistic)
            self.model.optimal_thresholds[i]=thresholds[optimal_threshold_index]-0.1
            self.model.optimal_thresholds[i]=thresholds[optimal_threshold_index]


            # Plotting
            # Plot Line with optimal threshold in legend
            plt.plot(fpr, tpr, label=CLASSES[i] + ' (AUC = %0.2f, Optimal Threshold = %0.2f)' % (auc, self.model.optimal_thresholds[i]), linewidth=2)

            # Store AUC
            AUCs[CLASSES[i]]=auc
        
    
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
        self.tensor_board_writer.add_image(f'Evaluation_Metrics/ROC_curve', image, global_step=0,dataformats='HWC')
        logging.info("ROC Added To Tensor board")

    
        return AUCs

      
    


    def compute_classification_metrics(self, y_true, y_pred, thresholds):
      '''
      Calculate classification metrics
      '''
      metrics={
        "false_positive":{},
        "false_negative":{},
        "true_positive":{},
        "true_negative":{},
        "precision":{},
        "recall":{},
        "f1":{},
      }

      best_thresholds=thresholds


      # # Find the best threshold for each class [Fine Tuning the threshold for each class]
      # thresholds=np.array(np.arange(0, 1.1, 0.1))
      # threshold_best=None
      # best_thresholds=[]
   

      # for i in range(len(CLASSES)):
      #   f1_best=-1
      #   for threshold in thresholds:
      #     fp = np.sum(np.logical_and(y_true[:, i] == 0, y_pred[:, i] >= threshold))
      #     fn = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] < threshold))
      #     tp = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] >= threshold))
      #     tn = np.sum(np.logical_and(y_true[:, i] == 0, y_pred[:, i] < threshold))

      #     precision = tp / (tp + fp)
      #     recall = tp / (tp + fn)
      #     f1 = 2 * (precision * recall) / (precision + recall)

      #     if f1>f1_best:
      #       threshold_best=threshold
      #       f1_best=f1
      #       print("f1_best",f1_best,"at threshold",threshold_best)
       
      #   best_thresholds.append(threshold_best)


      for i in range(len(CLASSES)):
          x = y_true[:, i]
          x[x<0] = 0
          x[x>1] =1
          y_true[:, i] = x 
          fp = np.sum(np.logical_and(y_true[:, i] == 0, y_pred[:, i] >= best_thresholds[i]))
          fn = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] < best_thresholds[i]))
          tp = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] >= best_thresholds[i]))
          tn = np.sum(np.logical_and(y_true[:, i] == 0, y_pred[:, i] < best_thresholds[i]))

          precision = tp / (tp + fp)
          recall = tp / (tp + fn)
          f1 = 2 * (precision * recall) / (precision + recall)

          metrics["false_positive"][CLASSES[i]]=fp
          metrics["false_negative"][CLASSES[i]]=fn
          metrics["true_positive"][CLASSES[i]]=tp
          metrics["true_negative"][CLASSES[i]]=tn

          metrics["precision"][CLASSES[i]]=precision
          metrics["recall"][CLASSES[i]]=recall
          metrics["f1"][CLASSES[i]]=f1
          
      return metrics

    def save_result_to_tensor_board(self,evaluation_loss,aucs,classification_metrics):
      # Loss
      self.tensor_board_writer.add_scalar('Evaluation_Metrics/evaluation_loss', evaluation_loss)

      # AUC
      self.tensor_board_writer.add_scalars('Evaluation_Metrics/auc', aucs)

      # classification metrics
      for metric,values in classification_metrics.items():
        self.tensor_board_writer.add_scalars(f'Evaluation_Metrics/{metric}', values)

    def error_analysis(self):
        '''
        Error Analysis
        '''
        # get best examples with lowest loss and worst examples with highest loss

        losses=[]
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx,(images,targets,_) in enumerate(self.data_loader_eval):
                logging.info(f"Batch {batch_idx}/{len(self.data_loader_eval)} ....")

                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE) 

                # Forward Pass [TODO]
                _,loss,scores=self.forward_pass(images,targets)
                
                evaluation_total_loss=loss

                # add loss to the list with the index
                losses.append((evaluation_total_loss,batch_idx))

                print(f"Batch {batch_idx}/{len(self.data_loader_eval)} Loss: {evaluation_total_loss}")

                del images
                del targets

            gc.collect()
            torch.cuda.empty_cache()       
        
        # Sort the losses
        losses.sort(key=lambda x: x[0], reverse=True)

        # Get the best and worst examples
        best_examples=[]
        worst_examples=[]
        for i in range(20):
            best_examples.append(losses[i])
            worst_examples.append(losses[-i-1])
        # save indices of best and worst examples
        best_indices=[index for loss,index in best_examples]
        worst_indices=[index for loss,index in worst_examples]

        # save it in file as numpy array
        np.save("best_examples.npy",best_indices)
        np.save("worst_examples.npy",worst_indices)
        

def init_working_space():

    # Creating tensorboard folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_board_folder_path="./tensor_boards/" + str(RUN)+ f"/eval_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")

    return tensor_board_folder_path

def main():
    logging.info(" Heat Map Evaluation Started")
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
        
    # Create an HeatMap Evaluation instance with the HeatMap model
    evaluator = HeatMapEvaluation(model=heat_map_model,tensor_board_writer=tensor_board_writer)

    # Start Evaluation
    evaluator.evaluate()        

if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/heat_map_u_ones_evaluator.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
    

     
# python -m src.heat_map_U_ones.evaluation.heat_map_evaluation