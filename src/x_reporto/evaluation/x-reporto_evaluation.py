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
import torchmetrics


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


    def compute_IOU(self,pred_box, target_box):
        '''
        Function to compute the Intersection over Union (IOU) of two boxes.

        inputs:

            pred_box: predicted box (Format [xmin, ymin, xmax, ymax])
            target_box: target box (Format [xmin, ymin, xmax, ymax])
        '''
        if pred_box is None or target_box is None:
            return 0

        # compute the intersection area
        x1 = max(pred_box[0], target_box[0])
        y1 = max(pred_box[1], target_box[1])
        x2 = min(pred_box[2], target_box[2])
        y2 = min(pred_box[3], target_box[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # compute the union area
        pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        target_box_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
        union_area = pred_box_area + target_box_area - intersection_area

        # compute the IOU 0 (no overlap) -> 1 totally overlap
        iou = intersection_area / union_area
        return iou

    def compute_confusion_metric(self,pred_boxes,pred_labels, target_boxes,target_labels, iou_threshold=0.5):
        '''
        Function to compute the precision.

        inputs:
            pred_boxes: list of predicted boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
            pred_labels: list of predicted labels (Format [N] => N times label)
            target_boxes: list of target boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
            target_labels: list of target labels (Format [N] => N times label)
            iou_threshold: threshold to consider a prediction to be correct
        '''
        # compute the number of true positive detections
        num_true_positive = 0
        num_false_positive = 0
        num_false_negative = 0
        index = 1
        # for each predicted box
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            # for each target box
            if pred_label != 0 and index in target_labels:
                if self.compute_IOU(pred_box, target_boxes[index-1]) > iou_threshold:
                    # increment the number of true positive detections
                    num_true_positive += 1
                else:
                    num_false_positive += 1
            elif pred_label != 0 and index not in target_labels:
                num_false_positive += 1
            elif pred_label == 0 and index in target_labels:
                num_false_negative += 1
            # increment the index
            index += 1            

        return num_true_positive, num_false_positive, num_false_negative

    def compute_confusion_metric_per_batch(self,pred_boxes,pred_labels, target_boxes,target_labels, iou_threshold=0.5):
        num_true_positive = 0
        num_false_positive = 0
        num_false_negative = 0
        for i in range(len(pred_boxes)):
            true_positive, false_positive, false_negative = self.compute_confusion_metric(pred_boxes[i], pred_labels[i], target_boxes[i], target_labels[i], iou_threshold)
            num_true_positive += true_positive
            num_false_positive += false_positive
            num_false_negative += false_negative
        logging.debug(f"True Positive: {num_true_positive}, False Positive: {num_false_positive}, False Negative: {num_false_negative}")
        return num_true_positive, num_false_positive, num_false_negative
    def evaluate(self):
        #validate the model
        if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value or MODEL_STAGE==ModelStage.CLASSIFIER.value:
            
            validation_total_loss,obj_detector_scores,region_selection_scores,region_abnormal_scores = self.validate_and_evalute_object_detection_and_classifier()
            
            print("validation_total_loss",validation_total_loss)
            print("obj_detector_scores avg_iou",obj_detector_scores["avg_iou"])
            print("region_selection_scores",region_selection_scores)
            print("region_abnormal_scores",region_abnormal_scores)
            sys.exit()
            correct_iou = obj_detector_scores["sum_iou_per_region"] / obj_detector_scores["sum_region_detected"]
            
            for region_indx, score in enumerate(correct_iou):
                # [Tensor Board]: Metric IOU
                self.tensor_board_writer.add_scalar(f'Evaluation_Metric_Object_Detector/Region_IOU',score,global_step=region_indx+1)

            # [Tensor Board]: Metric IOU
            # self.tensor_board_writer.add_scalar('Evaluation_Metric_Object_Detector/Average IOU',obj_detector_scores['avg_iou'],global_step=0)
            # [Tensor Board]: Metric Num_detected_regions_per_image
            self.tensor_board_writer.add_scalar('Evaluation_Metric_Object_Detector/Avgerage Num_detected_regions_per_image',obj_detector_scores['avg_num_detected_regions_per_image'],global_step=0)
            
            if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                # [Tensor Board]: Metric Region Selection
                self.tensor_board_writer.add_scalar(f'Evaluation_Metric_Abnormal_Classifier/F1-Score',region_abnormal_scores,global_step=0)
                self.tensor_board_writer.add_scalar(f'Evaluation_Metric_Abnormal_Classifier/Precision',region_abnormal_scores,global_step=0)
                self.tensor_board_writer.add_scalar(f'Evaluation_Metric_Abnormal_Classifier/Recall',region_abnormal_scores,global_step=0)

            for i,subset in enumerate(region_selection_scores):  
                    self.tensor_board_writer.add_scalar(f'Evaluation_Metric_Region_Selection_Classifier/F1-Score',region_selection_scores[subset]['f1'],global_step=i)
                    self.tensor_board_writer.add_scalar(f'Evaluation_Metric_Region_Selection_Classifier/Precision',region_selection_scores[subset]['precision'],global_step=i)
                    self.tensor_board_writer.add_scalar(f'Evaluation_Metric_Region_Selection_Classifier/Recall',region_selection_scores[subset]['recall'],global_step=i)
                                

    def update_region_abnormal_metrics(self,region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected):
        """
        Args:
            region_abnormal_scores (Dict)
            predicted_abnormal_regions (Tensor[bool]): shape [batch_size x 29]
            region_is_abnormal (Tensor[bool]): shape [batch_size x 29]
            class_detected (Tensor[bool]): shape [batch_size x 29]

        We only update/compute the scores for regions that were actually detected by the object detector (specified by class_detected).
        """
        detected_predicted_abnormal_regions = predicted_abnormal_regions[class_detected]
        detected_region_is_abnormal = region_is_abnormal[class_detected]

        region_abnormal_scores["precision"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
        region_abnormal_scores["recall"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
        region_abnormal_scores["f1"](detected_predicted_abnormal_regions, detected_region_is_abnormal) 

    def update_region_selection_metrics(self,region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal):
        """
        Args:
            region_selection_scores (Dict[str, Dict])
            selected_regions (Tensor[bool]): shape [batch_size x 29]
            region_has_sentence (Tensor[bool]): shape [batch_size x 29]
            region_is_abnormal (Tensor[bool]): shape [batch_size x 29]
        """
        normal_selected_regions = selected_regions[~region_is_abnormal]
        normal_region_has_sentence = region_has_sentence[~region_is_abnormal]

        abnormal_selected_regions = selected_regions[region_is_abnormal]
        abnormal_region_has_sentence = region_has_sentence[region_is_abnormal]

        region_selection_scores["all"]["precision"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))
        region_selection_scores["all"]["recall"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))
        region_selection_scores["all"]["f1"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))

        region_selection_scores["normal"]["precision"](normal_selected_regions, normal_region_has_sentence)
        region_selection_scores["normal"]["recall"](normal_selected_regions, normal_region_has_sentence)
        region_selection_scores["normal"]["f1"](normal_selected_regions, normal_region_has_sentence)

        region_selection_scores["abnormal"]["precision"](abnormal_selected_regions, abnormal_region_has_sentence)
        region_selection_scores["abnormal"]["recall"](abnormal_selected_regions, abnormal_region_has_sentence)
        region_selection_scores["abnormal"]["f1"](abnormal_selected_regions, abnormal_region_has_sentence)
    
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

            # calculate IOU using the intersection and union areas
            Iou = intersection_area / union_area

            # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
            Iou = torch.sum(Iou, dim=0)
                
            # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
            intersection_area = torch.sum(intersection_area, dim=0)
            union_area = torch.sum(union_area, dim=0)

            return intersection_area, union_area, Iou

        # sum up detections for each region
        region_detected_batch = torch.sum(class_detected, dim=0)

        intersection_area_per_region_batch, union_area_per_region_batch,iou_per_region = compute_intersection_and_union_area_per_region(detections, image_targets, class_detected)

        obj_detector_scores["sum_region_detected"] += region_detected_batch
        obj_detector_scores["sum_intersection_area_per_region"] += intersection_area_per_region_batch
        obj_detector_scores["sum_union_area_per_region"] += union_area_per_region_batch
        obj_detector_scores["sum_iou_per_region"] += iou_per_region
        # print iou
        # print("iou: ",iou_per_region)
    
    def draw_tensor_board(self,batch_idx,images,object_detector,region_selection_classifier,abnormal_region_classifier):
        # print(object_detector)

        object_detector_gold=object_detector['object_detector_targets']
        object_detector_boxes=object_detector['object_detector_boxes'].cpu()
        object_detector_detected_classes=object_detector['object_detector_detected_classes'].cpu()

        img_id=1
        # Draw Batch Images
        for i,image in enumerate(images):
            image=image.cpu()
            # Image
            # Plot Object Detector
            if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value :
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

            # if MODEL_STAGE==ModelStage.CLASSIFIER.value :
            #     # Region Selection Classifier
            #     region_selection_classifier_targets=region_selection_classifier['targets']
            #     region_selection_classifier_prediction=region_selection_classifier['predicted']
            #     region_selection_plot=plot_image_NEW(image,None,
            #                                          object_detector_gold[i]['labels'].cpu().tolist() ,object_detector_gold[i]['boxes'].cpu().tolist(),
            #                                          region_selection_classifier_targets.cpu().tolist(),
                                                    
            #                                          object_detector_detected_classes[i].tolist(),object_detector_boxes[i].tolist(),
            #                                          region_selection_classifier_prediction.cpu().tolist())
                
            #     # TODO Fix LIke PLot above
            #     region_selection_tensor=region_selection_plot
            #     # [Tensor Board]: Evaluation Image With Boxes
            #     self.tensor_board_writer.add_image(f'/Region Selection Classifier/'+str(batch_idx)+'_'+str(img_id), region_selection_tensor, global_step=0)

            #     # Upnormal Selection Classifier
            #     abnormal_region_classifier_targets=abnormal_region_classifier['targets']
            #     abnormal_region_classifier_prediction=abnormal_region_classifier['predicted']

            #     abnormal_region_plot=plot_image_NEW(image,None,
            #                                          object_detector_gold[i]['labels'].cpu().tolist() ,object_detector_gold[i]['boxes'].cpu().tolist(),
            #                                          abnormal_region_classifier_targets.cpu().tolist(),
            #                                          object_detector_detected_classes[i].tolist(),object_detector_boxes[i].tolist(),
            #                                          abnormal_region_classifier_prediction.cpu().tolist())


            #     # TODO Fix LIke PLot above
            #     abnormal_region_tensor=abnormal_region_plot
            #     # [Tensor Board]: Evaluation Image With Boxes
            #     self.tensor_board_writer.add_image(f'/Abnormal Classifier/'+str(batch_idx)+'_'+str(img_id), abnormal_region_tensor, global_step=0)

            # Increment Image Id
            img_id+=1

    
    def initalize_scorces(self):
        '''
        initalize_scorces
        '''
        obj_detector_scores = {}
        obj_detector_scores["sum_intersection_area_per_region"] = torch.zeros(29, device=DEVICE)
        obj_detector_scores["sum_union_area_per_region"] = torch.zeros(29, device=DEVICE)
        obj_detector_scores["sum_region_detected"] = torch.zeros(29, device=DEVICE)
        obj_detector_scores["sum_iou_per_region"] = torch.zeros(29, device=DEVICE)
        
        region_selection_scores = {}
        region_abnormal_scores = {}
        if MODEL_STAGE==ModelStage.CLASSIFIER.value:
            region_selection_scores = {}
            for subset in ["all", "normal", "abnormal"]:
                region_selection_scores[subset] = {
                    "precision": torchmetrics.Precision(num_classes=2, average=None,task='binary').to(DEVICE),
                    "recall": torchmetrics.Recall(num_classes=2, average=None,task='binary').to(DEVICE),
                    "f1": torchmetrics.F1Score(num_classes=2, average=None,task='binary').to(DEVICE),
                }

            region_abnormal_scores = {
                "precision": torchmetrics.Precision(num_classes=2, average=None,task='binary').to(DEVICE),
                "recall": torchmetrics.Recall(num_classes=2, average=None,task='binary').to(DEVICE),
                "f1": torchmetrics.F1Score(num_classes=2, average=None ,task='binary').to(DEVICE),
            }
        
        return obj_detector_scores,region_selection_scores,region_abnormal_scores
    
    def validate_and_evalute_object_detection_and_classifier(self):
        '''
        validate_during_evalute_object_detection_and_classifier
        '''
        obj_detector_scores , region_selection_scores , region_abnormal_scores = self.initalize_scorces()
        
        obj_detector_scores["true_positive"] = 0
        obj_detector_scores["false_positive"] = 0
        obj_detector_scores["false_negative"] = 0
        self.model.eval()
        with torch.no_grad():
            # validate the model
            logging.info("Validation the model")
            validation_total_loss=0
            for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,_,_) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                object_detector_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in object_detector_targets]
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
                region_selection_classifier={
                    "targets":selection_classifier_targets,
                    "predicted":selected_regions
                }

                abnormal_region_classifier={
                    "targets":abnormal_classifier_targets,
                    "predicted":predicted_abnormal_regions
                }
                self.draw_tensor_board(batch_idx,images,object_detector,region_selection_classifier,abnormal_region_classifier)


                validation_total_loss+=Total_loss
                
                #evaluate the model
                logging.info("Evaluating the model")
                # update scores for object detector metrics
                self.update_object_detector_metrics(obj_detector_scores, object_detector_boxes, object_detector_targets, object_detector_detected_classes)
                    # compute the confusion metric
                true_positive, false_positive, false_negative = self.compute_confusion_metric_per_batch(object_detector_boxes, object_detector_detected_classes, object_detector_targets[0]['boxes'], object_detector_targets[0]['labels'])
                obj_detector_scores["true_positive"] += true_positive
                obj_detector_scores["false_positive"] += false_positive
                obj_detector_scores["false_negative"] += false_negative

                if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                    # update scores for region selection metrics
                    self.update_region_selection_metrics(region_selection_scores, selected_regions, selection_classifier_targets , abnormal_classifier_targets)
                    # update scores for region abnormal detection metrics
                    self.update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, abnormal_classifier_targets, object_detector_detected_classes)
                
                logging.debug(f"True Positive: {obj_detector_scores["true_positive"]}, False Positive: {obj_detector_scores["false_positive"]}, False Negative: {obj_detector_scores["false_negative"]}")
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
            
            if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                # compute the "micro" average scores for region_selection_scores
                # only report results for the positive class (hence [1])
                for subset in region_selection_scores:
                    for metric, score in region_selection_scores[subset].items():
                        region_selection_scores[subset][metric] = score.compute().item()  

                # compute the "micro" average scores for region_abnormal_scores
                for metric, score in region_abnormal_scores.items():
                    region_abnormal_scores[metric] = score.compute().item()
                
            return validation_total_loss,obj_detector_scores,region_selection_scores,region_abnormal_scores
        
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