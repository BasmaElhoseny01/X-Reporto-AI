# Logging
from logger_setup import setup_logging
import logging


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

from config import RUN,PERIODIC_LOGGING,log_config
from config import *


class XReportoValidation():
    def __init__(self, model:XReporto,validation_csv_path:str = validation_csv_path):
        '''
        X-Reporto Validation Class
        Args:
        model: X-Reporto Model
        validation_csv_path: Path to the validation csv file
        ''' 
        self.model = model
        self.model.to(DEVICE)
        self.validation_csv_path = validation_csv_path
        logging.info("Train dataset loaded")

        self.data_loader_val = DataLoader(dataset=CustomDataset(self.validation_csv_path), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    def Validate(self):
        '''
        Evaluate the X-Reporto model on the validation dataset
        '''
        # make model in training mode
        logging.info("Start Validation")
        self.model.eval()
        with torch.no_grad():
            for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets) in enumerate(self.data_loader_val):                
                # Move inputs to Device
                images = images.to(DEVICE)
                object_detector_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in object_detector_targets]
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
                else:
                    Total_loss=self.object_detector_and_classifier_forward_pass(batch_idx=batch_idx,images=images,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets)
            # Free GPU memory 
            del Total_loss
            torch.cuda.empty_cache()
            gc.collect()  
        logging.info("Vaildation Done")
        
    def language_model_forward_pass(self,images:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor,object_detector_targets:torch.Tensor,selection_classifier_targets:torch.Tensor,abnormal_classifier_targets:torch.Tensor,LM_targets:torch.Tensor,batch_idx:int,loopLength:int,LM_Batch_Size:int):
        for batch in range(BATCH_SIZE):
            total_LM_losses=0
            for i in range(0,loopLength,LM_Batch_Size):
                
                # Forward Pass
                object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,stop= self.model(images,input_ids,attention_mask, object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_targets,batch,i)

                if stop:
                    break
                # Backward pass
                Total_loss=None
                object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                Total_loss=object_detector_losses_summation.clone()
                Total_loss+=selection_classifier_losses
                Total_loss+=abnormal_binary_classifier_losses
                Total_loss+=LM_losses
                total_LM_losses+=LM_losses

            logging.debug(f'Batch {batch_idx + 1}/{len(self.data_loader_val)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} LM_losses: {total_LM_losses:.4f} total_Loss: {object_detector_losses_summation+selection_classifier_losses+abnormal_binary_classifier_losses+total_LM_losses:.4f}')
            # Free GPU memory
            del LM_losses
            del object_detector_losses
            del selection_classifier_losses
            del abnormal_binary_classifier_losses
            torch.cuda.empty_cache()
            gc.collect()
        return Total_loss

    def  object_detector_and_classifier_forward_pass(self,batch_idx:int,images:torch.Tensor,object_detector_targets:torch.Tensor,selection_classifier_targets:torch.Tensor,abnormal_classifier_targets:torch.Tensor):

            # Forward Pass
            object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses= self.model(images,None,None, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets,None)
            
            # Backward pass
            Total_loss=None
            object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
            Total_loss=object_detector_losses_summation.clone()
            if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                Total_loss+=selection_classifier_losses
                Total_loss+=abnormal_binary_classifier_losses

            logging.debug(f'Batch {batch_idx + 1}/{len(self.data_loader_val)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f}  total_Loss: {Total_loss:.4f}')
            # Free GPU memory
            del LM_losses
            del object_detector_losses
            del selection_classifier_losses
            del abnormal_binary_classifier_losses
            torch.cuda.empty_cache()
            gc.collect()
            return Total_loss

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



def main():
    
    logging.info(" X_Reporto Started")
    # Logging Configurations
    log_config()
    if OperationMode.VALIDATION.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Validation Mode")
   
    # X-Reporto Trainer Object
    x_reporto_model = XReporto().create_model()

    # Create an XReportoTrainer instance with the X-Reporto model
    validator = XReportoValidation(model=x_reporto_model)


    # # Start Training
    validator.Validate()
        

if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/x_reporto_validator.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
    

     