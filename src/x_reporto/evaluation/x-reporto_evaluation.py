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

# Modules
from src.x_reporto.models.x_reporto_factory import XReporto
from src.x_reporto.data_loader.custom_dataset import CustomDataset

# Utils 
from src.utils import plot_image
from config import RUN,PERIODIC_LOGGING,log_config
from config import *
from torch.utils.tensorboard import SummaryWriter



class XReportoEvaluation():
    def __init__(self, model:XReporto,evaluation_csv_path:str = evaluation_csv_path,tensor_board_writer:SummaryWriter=None):
        '''
        X-Reporto Validation Class
        Args:
        model: X-Reporto Model
        evaluation_csv_path: Path to the validation csv file
        ''' 
        self.model = model
        self.model.to(DEVICE)
        self.evaluation_csv_path = evaluation_csv_path
        self.data_loader_val = DataLoader(dataset=CustomDataset(self.evaluation_csv_path), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)
        self.tensor_board_writer=tensor_board_writer
        logging.info("Evalution dataset loaded")



    def evaluate(self):
        #validate the model
        if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value or MODEL_STAGE==ModelStage.CLASSIFIER.value:
            validation_total_loss,obj_detector_scores,_,_,_ = self.validate_during_evalute_object_detection_and_classifier()
            # print(f"Validation Total Loss: {validation_total_loss:.4f}")
            # print(f"Average IOU: {obj_detector_scores['avg_iou']:.4f}")

            # [Tensor Board]: Metric IOU
            self.tensor_board_writer.add_scalar('Evaluation_Metric_Object_Detector/Average IOU',obj_detector_scores['avg_iou'],global_step=0)
            # [Tensor Board]: Metric Num_detected_regions_per_image
            self.tensor_board_writer.add_scalar('Evaluation_Metric_Object_Detector/Avgerage Num_detected_regions_per_image',obj_detector_scores['avg_num_detected_regions_per_image'],global_step=0)


        

    def update_object_detector_metrics(self,obj_detector_scores, detections, image_targets, class_detected):
        def compute_box_area(box):
            """
            Calculate the area of a box given the 4 corner values.

            Args:
                box (Tensor[batch_size x 29 x 4])

            Returns:
                area (Tensor[batch_size x 29])
            """
            x0 = box[..., 0]
            y0 = box[..., 1]
            x1 = box[..., 2]
            y1 = box[..., 3]

            return (x1 - x0) * (y1 - y0)

        def compute_intersection_and_union_area_per_region(detections, targets, class_detected):
            # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
            # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
     
            pred_boxes = detections
            # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
            # gt_boxes is of shape [batch_size x 29 x 4]
            gt_boxes = torch.stack([t for t in targets[0]['boxes']], dim=0)
            # print("gt_boxes[..., 0]",gt_boxes[..., 0])
            # print("pred_boxes[..., 0]",pred_boxes[..., 0])
            # below tensors are of shape [batch_size x 29]
            x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
            y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
            x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
            y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])

            # intersection_boxes is of shape [batch_size x 29 x 4]
            intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

            # below tensors are of shape [batch_size x 29]
            intersection_area = compute_box_area(intersection_boxes)
            pred_area = compute_box_area(pred_boxes)
            gt_area = compute_box_area(gt_boxes)

            # if x0_max >= x1_min or y0_max >= y1_min, then there is no intersection
            valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

            # also there is no intersection if the class was not detected by object detector
            valid_intersection = torch.logical_and(valid_intersection, class_detected)

            # set all non-valid intersection areas to 0
            intersection_area[~valid_intersection] = 0

            union_area = (pred_area + gt_area) - intersection_area

            # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
            intersection_area = torch.sum(intersection_area, dim=0)
            union_area = torch.sum(union_area, dim=0)

            return intersection_area, union_area

        # sum up detections for each region
        region_detected_batch = torch.sum(class_detected, dim=0)

        intersection_area_per_region_batch, union_area_per_region_batch = compute_intersection_and_union_area_per_region(detections, image_targets, class_detected)

        obj_detector_scores["sum_region_detected"] += region_detected_batch
        obj_detector_scores["sum_intersection_area_per_region"] += intersection_area_per_region_batch
        obj_detector_scores["sum_union_area_per_region"] += union_area_per_region_batch
    
    def draw_tensor_board(self,batch_idx,images,object_detector):
        # print(object_detector)

        object_detector_gold=object_detector['object_detector_targets']
        object_detector_boxes=object_detector['object_detector_boxes'].cpu()
        object_detector_detected_classes=object_detector['object_detector_detected_classes'].cpu()

        img_id=1
        # Draw Batch Images
        for i,image in enumerate(images):
            image=image.cpu()
            # Ima
            regions=plot_image(image,None,object_detector_gold[i]['labels'].cpu().tolist() ,object_detector_gold[i]['boxes'].cpu().tolist(),object_detector_detected_classes[i].tolist(),object_detector_boxes[i].tolist())
            for j,region in enumerate(regions):
           
                # convert region to tensor
                region = region.astype(np.uint8)

                # convert numpy array to PyTorch tensor
                region_tensor = torch.from_numpy(region)

                # make sure the tensor has the shape (C, H, W)
                region_tensor = region_tensor.permute(2, 0, 1)
                # [Tensor Board]: Evaluation Image With Boxes
                self.tensor_board_writer.add_image(f'/Object Detector/'+str(batch_idx)+'_'+str(img_id), region_tensor, global_step=j+1)

            img_id+=1



    def validate_during_evalute_object_detection_and_classifier(self):
        '''
        validate_during_evalute_object_detection_and_classifier
        '''
        obj_detector_scores = {}
        obj_detector_scores["sum_intersection_area_per_region"] = torch.zeros(29, device=DEVICE)
        obj_detector_scores["sum_union_area_per_region"] = torch.zeros(29, device=DEVICE)
        obj_detector_scores["sum_region_detected"] = torch.zeros(29, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            # validate the model
            logging.info("Validation the model")
            validation_total_loss=0
            for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets) in enumerate(self.data_loader_val):
                # Move inputs to Device
                print("hreeee")
                images = images.to(DEVICE)
                object_detector_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in object_detector_targets]
                # if object_detector_targets[0]['boxes'].shape[0] != 29:
                #     continue
                if MODEL_STAGE==ModelStage.CLASSIFIER.value :
                    # Selection Classifier
                    # Moving Selection Classifier Targets to Device
                    selection_classifier_targets = selection_classifier_targets.to(DEVICE)
                    abnormal_classifier_targets = abnormal_classifier_targets.to(DEVICE)
                Total_loss,object_detector_boxes,object_detector_detected_classes,selected_regions,predicted_abnormal_regions=self.object_detector_and_classifier_forward_pass(batch_idx=batch_idx,images=images,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets)
                
                # [Tensor Board]
                object_detector={
                    "object_detector_targets":object_detector_targets,
                    "object_detector_boxes":object_detector_boxes,
                    "object_detector_detected_classes":object_detector_detected_classes,

                }
                self.draw_tensor_board(batch_idx,images,object_detector)

                validation_total_loss+=Total_loss
                #evaluate the model
                logging.info("Evaluating the model")

                self.update_object_detector_metrics(obj_detector_scores, object_detector_boxes, object_detector_targets, object_detector_detected_classes)
            
            # arverge validation_total_loss
            validation_total_loss/=(len(self.data_loader_val))

            # compute object detector scores
            sum_intersection = obj_detector_scores["sum_intersection_area_per_region"]
            sum_union = obj_detector_scores["sum_union_area_per_region"]
            obj_detector_scores["avg_iou"] = (torch.sum(sum_intersection) / torch.sum(sum_union)).item()
            obj_detector_scores["avg_iou_per_region"] = (sum_intersection / sum_union).tolist()
            sum_region_detected = obj_detector_scores["sum_region_detected"]
            obj_detector_scores["avg_num_detected_regions_per_image"] = torch.sum(sum_region_detected / len(self.data_loader_val)).item()
            obj_detector_scores["avg_detections_per_region"] = (sum_region_detected / len(self.data_loader_val)).tolist()

            return validation_total_loss,obj_detector_scores,None,None,None
        
    def language_model_forward_pass(self,images:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor,object_detector_targets:torch.Tensor,selection_classifier_targets:torch.Tensor,abnormal_classifier_targets:torch.Tensor,LM_targets:torch.Tensor,batch_idx:int,loopLength:int,LM_Batch_Size:int):
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
            
            # Backward pass
            Total_loss=None
            object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
            Total_loss=object_detector_losses_summation.clone()
            if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                Total_loss+=selection_classifier_losses
                Total_loss+=abnormal_binary_classifier_losses

            logging.debug(f'Batch {batch_idx + 1}/{len(self.data_loader_val)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f}  total_Loss: {Total_loss:.4f}')
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
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_board_folder_path="./tensor_boards/" + str(RUN)+ f"/eval_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")

    return tensor_board_folder_path

def main():
    
    logging.info(" X_Reporto Started")
    # Logging Configurations
    log_config()
    if OperationMode.EVALUATION.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Evaluation Mode")
    
    # Tensor Board
    tensor_board_folder_path=init_working_space()
    tensor_board_writer=SummaryWriter(tensor_board_folder_path)

    # X-Reporto Trainer Object
    x_reporto_model = XReporto().create_model()

    # Create an XReportoTrainer instance with the X-Reporto model
    evaluator = XReportoEvaluation(model=x_reporto_model,tensor_board_writer=tensor_board_writer)

    # Start Training
    evaluator.evaluate()
        

if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/x_reporto_Evaluator.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
    

     
# python -m src.x_reporto.evaluation.x-reporto_evaluation