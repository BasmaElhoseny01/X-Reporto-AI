# Logging
from logger_setup import setup_logging
import logging

from torch.utils.tensorboard import SummaryWriter

import os
import gc

# Torch
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader

# Modules
from src.x_reporto.models.x_reporto_factory import XReporto
from src.x_reporto.data_loader.custom_dataset import CustomDataset

# Utils 
from src.utils import empty_folder
from config import RUN,PERIODIC_LOGGING,log_config
from config import *


class XReportoTesting():
    def __init__(self, model:XReporto,tensor_board_writer:SummaryWriter,test_csv_path:str = test_csv_path):
        '''
        X-Reporto test Class
        Args:
        model: X-Reporto Model
        test_csv_path: Path to the test csv file
        ''' 
        self.model = model

        self.tensor_board_writer=tensor_board_writer

        # Move to device
        self.model.to(DEVICE)
        
        self.test_csv_path = test_csv_path

        # create dataset
        self.dataset_test= CustomDataset(self.test_csv_path)
        logging.info("Test dataset loaded")

        # Create Dataloader
        self.data_loader_test = DataLoader(dataset=self.dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
        logging.info(f"Test dataLoader loaded Size: {len(self.data_loader_test)}")

    def test(self):
        '''
        Evaluate the X-Reporto model on the Testing dataset
        '''
        # make model in training mode
        logging.info("Start Testing")
        self.model.eval()
        with torch.no_grad():
            for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets) in enumerate(self.data_loader_test):                
                # Move inputs to Device
                images = images.to(DEVICE)
                object_detector_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in object_detector_targets]
                #   
                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Selection Classifier
                    # Moving Selection Classifier Targets to Device
                    selection_classifier_targets = selection_classifier_targets.to(DEVICE)
                    abnormal_classifier_targets = abnormal_classifier_targets.to(DEVICE)
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    # Language Model
                    # Moving Language Model Targets to Device
                    LM_targets = LM_targets.to(DEVICE)
                    input_ids = LM_inputs['input_ids'].to(DEVICE)
                    attention_mask = LM_inputs['attention_mask'].to(DEVICE)
                    loopLength= input_ids.shape[1]
                    Total_loss=self.language_model_forward_pass(batch_idx=batch_idx,images=images,input_ids=input_ids,attention_mask=attention_mask,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets,LM_targets=LM_targets,loopLength=loopLength,LM_Batch_Size=LM_Batch_Size)
                elif ModelStage.CLASSIFIER.value==MODEL_STAGE or ModelStage.OBJECT_DETECTOR.value==MODEL_STAGE:
                   
                    Total_loss,object_detector_boxes,object_detector_detected_classes,selected_regions,predicted_abnormal_regions=self.object_detector_and_classifier_forward_pass(batch_idx=batch_idx,images=images,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets)
                    self.computer_model_metrics(object_detector_boxes,object_detector_detected_classes,selected_regions,predicted_abnormal_regions)
                    
            # Free GPU memory 
            del Total_loss
            torch.cuda.empty_cache()
            gc.collect()  
        logging.info("Testing Done")

    def compute_model_metrics(object_detector_boxes,object_detector_detected_classes,selected_regions,predicted_abnormal_regions):
        # Calculate Metrics
        # TensorBoard View
        logging.error("computer_model_metrics() Not implemented Yet")
    
    def language_model_forward_pass(self,images:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor,object_detector_targets:torch.Tensor,selection_classifier_targets:torch.Tensor,abnormal_classifier_targets:torch.Tensor,LM_targets:torch.Tensor,batch_idx:int,loopLength:int,LM_Batch_Size:int):
        logging.error("language_model_forward_pass() Not implemented Yet")
        pass
        # for batch in range(BATCH_SIZE):
        #     total_LM_losses=0
        #     for i in range(0,loopLength,LM_Batch_Size):
                
        #         # Forward Pass
        #         object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,stop= self.model(images,input_ids,attention_mask, object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_targets,batch,i)

        #         if stop:
        #             break
        #         # Backward pass
        #         Total_loss=None
        #         object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
        #         Total_loss=object_detector_losses_summation.clone()
        #         Total_loss+=selection_classifier_losses
        #         Total_loss+=abnormal_binary_classifier_losses
        #         Total_loss+=LM_losses
        #         total_LM_losses+=LM_losses

        #     logging.debug(f'Batch {batch_idx + 1}/{len(self.data_loader_val)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} LM_losses: {total_LM_losses:.4f} total_Loss: {object_detector_losses_summation+selection_classifier_losses+abnormal_binary_classifier_losses+total_LM_losses:.4f}')
        #     # Free GPU memory
        #     del LM_losses
        #     del object_detector_losses
        #     del selection_classifier_losses
        #     del abnormal_binary_classifier_losses
        #     torch.cuda.empty_cache()
        #     gc.collect()
        # return Total_loss

    def  object_detector_and_classifier_forward_pass(self,batch_idx:int,images:torch.Tensor,object_detector_targets:torch.Tensor,selection_classifier_targets:torch.Tensor,abnormal_classifier_targets:torch.Tensor):

            # Forward Pass
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions,_,_,_= self.model(images,None,None, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets,None)
            
            # In object Detector Mode
            if selection_classifier_losses is None : selection_classifier_losses=0.0
            if abnormal_binary_classifier_losses is None : abnormal_binary_classifier_losses=0.0

            # Backward pass
            Total_loss=None
            object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
            Total_loss=object_detector_losses_summation.clone()
            if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                Total_loss+=selection_classifier_losses
                Total_loss+=abnormal_binary_classifier_losses
           
            logging.debug(f'Batch {batch_idx + 1}/{len(self.dataset_test)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f}  total_Loss: {Total_loss:.4f}')            
           
            # [Tensor Board]: Object Detector Avg Batch Loss
            self.tensor_board_writer.add_scalar('Object Detector Avg Batch Loss',object_detector_losses_summation,batch_idx)
            # [Tensor Board]: Total Batch Loss
            self.tensor_board_writer.add_scalar('Avg Batch Total Losses',Total_loss,batch_idx)


            # Free GPU memory
            del object_detector_losses
            del selection_classifier_losses
            del abnormal_binary_classifier_losses
            torch.cuda.empty_cache()
            gc.collect()
            return Total_loss,object_detector_boxes,object_detector_detected_classes,selected_regions,predicted_abnormal_regions

def collate_fn(batch):
        image_shape = batch[0][0]["image"].size()
        images = torch.empty(size=(len(batch), *image_shape))
        object_detector_targets=[]
        selection_classifier_targets=[]
        abnormal_classifier_targets=[]
        LM_targets=[]
        input_ids=[]
        attention_mask=[]
        LM_inputs={}

        for i in range(len(batch)):
            (object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) = batch[i]
            # stack images
            images[i] = object_detector_batch['image']
            # Moving Object Detector Targets to Device
            new_dict={}
            new_dict['boxes']=object_detector_batch['bboxes']
            new_dict['labels']=object_detector_batch['bbox_labels']
            object_detector_targets.append(new_dict)
            
            bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal']
            abnormal_classifier_targets.append(bbox_is_abnormal)

            phrase_exist=selection_classifier_batch['bbox_phrase_exists']
            selection_classifier_targets.append(phrase_exist)

            phrase=LM_batch['label_ids']
            LM_targets.append(phrase)
            input_ids.append(LM_batch['input_ids'])
            attention_mask.append(LM_batch['attention_mask'])


        selection_classifier_targets=torch.stack(selection_classifier_targets)
        abnormal_classifier_targets=torch.stack(abnormal_classifier_targets)
        LM_targets=torch.stack(LM_targets)
        LM_inputs['input_ids']=torch.stack(input_ids)
        LM_inputs['attention_mask']=torch.stack(attention_mask)

        return images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets


def init_working_space():

    # Creating tensorboard folder
    tensor_board_folder_path="./tensor_boards" + str(RUN)+ "/test"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")
        empty_folder(tensor_board_folder_path)

    return tensor_board_folder_path


def main():    
    logging.info("Testing X_Reporto Started")

    # Logging Configurations
    log_config()
    if OperationMode.TESTING.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Testing Mode")

    if TRAIN_ROI :
        raise Exception("TRAIN_ROI Must be False")
    
    if TRAIN_RPN :
        raise Exception("TRAIN_RPN Must be False")

    tensor_board_folder_path=init_working_space()

   
    # X-Reporto Object
    x_reporto_model = XReporto().create_model()

    # Tensor Board
    tensor_board_writer=SummaryWriter(tensor_board_folder_path)

    # Create an XReportoTesting instance with the X-Reporto model
    tester = XReportoTesting(model=x_reporto_model,tensor_board_writer=tensor_board_writer)


    # Start Training
    tester.test()
        

if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/x_reporto_Tester.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
    

# python -m src.x_reporto.testing.X_reporto_testing.py