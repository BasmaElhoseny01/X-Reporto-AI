import sys
import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional, List, Dict

from config import ModelStage,MODEL_STAGE,DEVICE

# Modules
from src.object_detector.models.object_detector_factory import ObjectDetector

from src.binary_classifier.models.binary_classifier_selection_region_factory import BinaryClassifierSelectionRegion
from src.binary_classifier.models.binary_classifier_region_abnormal_factory import BinaryClassifierRegionAbnormal
from src.language_model.GPT2.gpt2_model import CustomGPT2
from src.language_model.GPT2.config import Config
class XReportoV1(nn.Module):
    """
    A modular model for object detection and binary classification.

    Attributes:
        - num_classes (int): The number of classes for object detection.

    Modules:
        - object_detector (ObjectDetectorWrapper): Object detector module.
        - binary_classifier_selection_region (BinaryClassifierSelectionRegionWrapper): Binary classifier for region selection.
        - binary_classifier_region_abnormal (BinaryClassifierRegionAbnormalWrapper): Binary classifier for abnormal region detection.
    """

    def __init__(self):
        super().__init__()
        self.num_classes=30

        self.object_detector = ObjectDetector().create_model()

        if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
            self.binary_classifier_selection_region = BinaryClassifierSelectionRegion().create_model()
            self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal().create_model()
        if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
            config = Config()
            config.d_model = 768
            config.d_ff1 = 768
            config.d_ff2 = 768
            config.d_ff3 = 768
            config.num_layers = 12
            config.vocab_size = 50257
            config.max_seq_len = 1024
            config.pretrained_model = "gpt2"
            image_config = Config()
            image_config.d_model = 1024
            image_config.d_ff1 = 1024
            image_config.d_ff2 = 1024
            image_config.d_ff3 = 768
            image_config.num_heads = 8
            image_config.num_layers = 6
            image_config.vocab_size = 50257
            image_config.max_seq_len = 1024
            image_config.dropout = 0.1
            self.language_model = CustomGPT2(config,image_config)

    def forward(self,images: Tensor ,input_ids=None,attention_mask=None, object_detector_targets: Optional[List[Dict[str, Tensor]]] = None, selection_classifier_targets: Tensor=None,abnormal_classifier_targets: Tensor = None,language_model_targets: Tensor= None,):
        '''
        Forward pass through the X-ReportoV1 model.

        Args:
            - images (Tensor): Input images of shape [batch_size x 1 x 512 x 512] (grey-scaled images) [Normalized 0-1].
            - object_detector_targets (Optional[List[Dict[str, Tensor]]]): List of dictionaries containing bounding box targets for each batch element
                Each dictionary in the list should have the following keys:
                - 'boxes' (FloatTensor[N, 4]): Ground-truth boxes in [x1, y1, x2, y2] format, where N is the number of bounding boxes detected.
                - 'labels' (Int64Tensor[N]): Ground-truth labels for each box, where N is the number of bounding boxes detected.
                If None, the model assumes inference mode without ground truth labels.

            - selection_classifier_targets (Optional[Tensor]):Binary Tensor of shape [batch_size x,29]
                Ground truth indicating whether a phrase exists in the region or not.
                In CLASSIFIER Mode:
                    If None , the model assumes inference mode without ground truth labels.

            - abnormal_classifier_targets (Optional[Tensor]):Binary Tensor of shape [batch_size x,29]
                Ground truth indicating whether a region is abnormal or not.
                In CLASSIFIER Mode:
                    If None , the model assumes inference mode without ground truth labels.

            - language_model__targets

        Returns:
            If in OBJECT_DETECTOR Mode:
                If in training mode:
                 - object_detector_losses Dict: Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                - selection_classifier_losses: 0
                - abnormal_binary_classifier_losses: 0

                If in Validation mode:
                - object_detector_losses Dict: Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                    Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                    Indicates if the object detector has detected the region/class or not.

                If in Inference mode:           
                    - object_detector_losses (Dict): Empty Dictionary
                    - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                        Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                    - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                        Indicates if the object detector has detected the region/class or not.


            If in CLASSIFIER Mode:
                If in training mode:
                    - object_detector_losses (Dict): Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                   - selection_classifier_losses (Tensor): Loss of the Selection Region Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                   - abnormal_binary_classifier_losses (Tensor): Loss of the Abnormal Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.

                If in Validation mode:
                    - object_detector_losses (Dict): Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                    - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                        Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                    - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                        Indicates if the object detector has detected the region/class or not.

                    - selection_classifier_losses(Tensor): Loss of the Selection Region Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - selected_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                        Indicating selected regions

                    - abnormal_binary_classifier_losses(Tensor): Loss of the Abnormal Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                       Indicating predicted abnormal regions.

                If in Inference mode:
                    - object_detector_losses (Dict): Empty Dictionary
                    - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                        Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                    - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                        Indicates if the object detector has detected the region/class or not.

                    - selection_classifier_losses: None.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - selected_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                        Indicating selected regions

                    - abnormal_binary_classifier_losses: None.
                    - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                        Indicating predicted abnormal regions.    
       '''
        if self.training:
            # Training
            # Stage(1) Object Detector
            print("Before object detector")
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
            del images
            del object_detector_targets
            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,0,0
            # Stage(2) Binary Classifier
            print("Before binary classifier selection region")
            object_detector_detected_classes=object_detector_detected_classes.to(DEVICE)
            selection_classifier_losses,_,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,_=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            del abnormal_classifier_targets
            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses
            
            # valid_input_ids, valid_attention_mask, valid_region_features=self.get_valid_decoder_input_for_training(object_detector_detected_classes, selection_classifier_targets, input_ids, attention_mask, object_detector_features)
            del selection_classifier_targets
          
            print("Before language model")
            LM_output=self.language_model(input_ids=input_ids,image_hidden_states=object_detector_features,attention_mask=attention_mask,labels=language_model_targets)
            del object_detector_features
            del input_ids
            del attention_mask
            return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_output[0]
       
        else: # Validation (or inference) mode
            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,object_detector_boxes,object_detector_detected_classes
            
            # Stage(2) Binary Classifier
            selection_classifier_losses,selected_regions,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,predicted_abnormal_regions=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            
            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions
    
    
    def get_valid_decoder_input_for_training(
        self,
        class_detected,  # shape [batch_size x 29]
        region_has_sentence,  # shape [batch_size x 29]
        input_ids,  # shape [(batch_size * 29) x seq_len]
        attention_mask,  # shape [(batch_size * 29) x seq_len]
        region_features,  # shape [batch_size x 29 x 1024]
    ):
        """
        We want to train the decoder only on region features (and corresponding input_ids/attention_mask) whose corresponding sentences are non-empty and
        that were detected by the object detector.
        """
        # valid is of shape [batch_size x 29]
        valid = torch.logical_and(class_detected, region_has_sentence)

        # reshape to [(batch_size * 29)], such that we can apply the mask to input_ids and attention_mask
        valid_reshaped = valid.reshape(-1)

        valid_input_ids = input_ids[valid_reshaped]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_attention_mask = attention_mask[valid_reshaped]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_region_features = region_features[valid]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x 1024]

        return valid_input_ids, valid_attention_mask, valid_region_features