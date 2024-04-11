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
from src.utils import load_model

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
               
        

    def F1_score_for_each_class(self, y_true, y_pred,thresholds):
        '''
        F1 Score
        '''
        # y_true = y_true.cpu().detach().numpy()
        # y_pred = y_pred.cpu().detach().numpy()
        f1_scores = []
        for i in range(len(CLASSES)):
            for j in range(len(y_pred[:, i])):
                if y_pred[j, i] >= thresholds[i]:
                    y_pred[j, i] = 1
                else:
                    y_pred[j, i] = 0

        for i in range(len(CLASSES)):
            false_positive = np.sum(np.logical_and(y_true[:, i] == 0, y_pred[:, i] == 1))
            false_negative = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] == 0))
            true_positive = np.sum(np.logical_and(y_true[:, i] == 1, y_pred[:, i] == 1))
            true_negative = np.sum(np.logical_and(y_true[:, i] == 0, y_pred[:, i] == 0))
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
            print(f'Class: {CLASSES[i]}, Precision: {precision}, Recall: {recall}, F1: {f1}')
            print(f'False Positive: {false_positive}, False Negative: {false_negative}, True Positive: {true_positive}, True Negative: {true_negative}')
        return f1_scores
    
    def evaluate(self):
        #Evaluate the model
        eval_loss = self.evaluate_heat_map()
        
        print("Evaluation Loss",eval_loss)
        
        # logging precision and recall

        # [Tensor Board] Update the Board by the scalers for that Run
        # self.update_tensor_board_score()
        

    def evaluate_heat_map(self):
        # Init The Scores
        heat_map_scores = self.initalize_scorces()
        all_preds= np.zeros((1, len(CLASSES)))
        all_targets= np.zeros((1, len(CLASSES)))
      
        
        self.model.eval()
        with torch.no_grad():
            # evaluate the model
            logging.info("Evaluating the model")
            
            for batch_idx,(images,targets,_) in enumerate(self.data_loader_eval):
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE) 

                # Forward Pass [TODO]
                _,loss,scores=self.forward_pass(images,targets)
                
                validation_total_loss=loss
                        
                # Cumulate all predictions ans labels
                all_preds = np.concatenate((all_preds, scores.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)
                all_targets = np.concatenate((all_targets, targets.to("cpu").detach().view(-1, len(CLASSES)).numpy()), 0)
                
                # Update Score
                # self.update_heat_map_metrics(heat_map_scores, classes, targets)
                
                # [Tensor Board] Draw the HeatMap Predictions of this batch
                #TODO: uncomment
                # self.draw_tensor_board(batch_idx,images,features,classes)

            # Compute ROC
            thresholds= self.compute_ROC(y_true=all_targets[1:,:],y_scores=all_preds[1:,:],n_classes=len(CLASSES))                
            # F1
            f1_scores = self.F1_score_for_each_class(all_targets[1:,:], all_preds[1:,:],thresholds)
            
            
        return validation_total_loss
    
    
    def forward_pass(self,images,targets):
        '''
        y: Prob not classes
        '''
        # Forward Pass
        y_pred,scores,_=self.model(images)

        # Calculate Loss
        Total_loss=nn.BCEWithLogitsLoss(reduction='mean')(y_pred,targets)*images[0].size(0)
        
        features=None
        
        return features,Total_loss,scores

    def initalize_scorces(self):
        heat_map_scores={key: {} for key in CLASSES}
        
        for disease in CLASSES:
            heat_map_scores[disease]['true_positive']=0
            heat_map_scores[disease]['false_positive']=0
            heat_map_scores[disease]['true_negative']=0
            heat_map_scores[disease]['false_negative']=0
        return heat_map_scores
    
    def compute_ROC(self,y_true,y_scores,n_classes):
        # print("y_scores",y_scores)
        # print("y_true",y_true)
        plt.figure(figsize=(10, 8))  # Adjust figure size

        optimal_thresholds=[]
    
        # Draw ROC Curve for Each Class
        for i in range(len(CLASSES)):    
            fpr, tpr, thresholds = metrics.roc_curve(y_true[:, i], y_scores[:, i])

            # AUC
            roc_auc = metrics.auc(fpr, tpr)
            
            # Compute Youden's J statistic
            j_statistic = tpr - fpr

            # Find the index of the threshold that maximizes J statistic
            optimal_threshold_index = np.argmax(j_statistic)

            # Get the optimal threshold
            optimal_threshold = thresholds[optimal_threshold_index]

            # Add Optimal Thresholds
            optimal_thresholds.append(optimal_threshold)

            # Plot Line with optimal threshold in legend
            plt.plot(fpr, tpr, label=CLASSES[i] + ' (AUC = %0.2f, Optimal Threshold = %0.2f)' % (roc_auc, optimal_threshold), linewidth=2)


        # Add legend, labels, and grid
        plt.legend(loc='lower right', fontsize=8)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.title('ROC Curves for Different Classes', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)  # Decrease tick label font size
        plt.grid(True)
        
        # [TODO] Add To Tensor board
        # Save figure as PNG image in memory buffer
        #buf = io.BytesIO()
        #plt.savefig(buf, format='png')
        #buf.seek(0)

        # Convert PNG image buffer to TensorFlow Summary
        #image = tf.image.decode_png(buf.getvalue(), channels=4)
        #image = tf.expand_dims(image, 0)

        # Write Summary to TensorBoard
        #with tf.summary.create_file_writer("./models/heat_map_4").as_default():
            #tf.summary.image("ROC Curve", image, step=0)  # Use appropriate step value

        #plt.close()

        # Save figure with appropriate DPI
        plt.savefig(f"./tensor_boards/heat_maps/{RUN}/roc.png", dpi=300)
        plt.show()

        return optimal_thresholds
        

def init_working_space():

    # Creating tensorboard folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_datetime="test"
    tensor_board_folder_path="./tensor_boards/" + "heat_maps/" + str(RUN)+ f"/eval_{current_datetime}"
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