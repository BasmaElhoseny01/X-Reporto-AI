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

from sklearn.metrics import roc_auc_score

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
        
        self.evaluation_csv_path = evaluation_csv_path
        self.tensor_board_writer=tensor_board_writer

        
        self.data_loader_val = DataLoader(dataset=HeatMapDataset(self.evaluation_csv_path), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        logging.info("Evaluation dataset loaded")
        print("Evaluation dataset loaded")
        
        
    def evaluate(self):
        #Evaluate the model
        scores = self.evaluate_heat_map()
        
        # logging precision and recall

        # [Tensor Board] Update the Board by the scalers for that Run
        # self.update_tensor_board_score()
        

    def evaluate_heat_map(self):
        # Init The Scores
        heat_map_scores = self.initalize_scorces()
        
        self.model.eval()
        with torch.no_grad():
            # validate the model
            logging.info("Evaluating the model")
            validation_total_loss=0
            
            pred_labels= torch.FloatTensor().cuda()
            gold_labels= torch.FloatTensor().cuda()
            
            for batch_idx,(images,targets) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                targets=targets.to(DEVICE) 

                # Forward Pass [TODO]
                features,Total_loss,classes=self.forward_pass(images,targets)
                
                gold_labels = torch.cat((gold_labels, targets), 0)
                pred_labels = torch.cat((pred_labels, classes), 0)
                
                # Update Score
                # self.update_heat_map_metrics(heat_map_scores, classes, targets)
                
                # [Tensor Board] Draw the HeatMap Predictions of this batch
                #TODO: uncomment
                # self.draw_tensor_board(batch_idx,images,features,classes)

                validation_total_loss+=Total_loss
                
            self.computer_AUROC(gold=gold_labels,pred=pred_labels,n_classes=13)
                
            
        return validation_total_loss

    def initalize_scorces(self):
        heat_map_scores={key: {} for key in CLASSES}
        
        for disease in CLASSES:
            heat_map_scores[disease]['true_positive']=0
            heat_map_scores[disease]['false_positive']=0
            heat_map_scores[disease]['true_negative']=0
            heat_map_scores[disease]['false_negative']=0
        return heat_map_scores
    
    def computer_AUROC(self,gold,pred,n_classes):
        outAUROC = []
            
        gold = gold.cpu().numpy()
        pred = pred.cpu().numpy()
#         print(gold[:,0])
#         print(pred[:,0])
#         sys.exit()
        
        for i in range(n_classes):
            try:
                outAUROC.append(roc_auc_score(gold[:, i], pred[:, i]))
            except ValueError:
                logging.info("Class Has One Value" + CLASSES[i])
                outAUROC.append(-1)
                pass
        
                
        print(outAUROC)
        sys.exit()
        
        pass
        
    
    def update_heat_map_metrics(self,heat_map_scores, predicted_classes, targets):
        for i in range(len(targets)):
            # Each Example in the Batch
#             for disease_idx,disease in range(CLASSES):
#                 if predicted_classes[i][disease_idx].item()
                
            print(predicted_classes[i])
            print(targets[i])
            print(predicted_classes.shape)
            print(targets.shape)
        sys.exit()
            
        
            
        
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
        # Forward Pass
        y=self.model(images)
        features=None

        # Calculate Loss
        #Total_loss=self.criterion(y,targets)
        Total_loss=torch.nn.BCELoss(reduction = 'mean').to(DEVICE)(y,targets)
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

    logging.info("Loading heat_map ....")
    load_model(model=heat_map_model,name='heat_map_best')
        
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