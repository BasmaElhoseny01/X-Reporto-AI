import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional, List, Dict

from config import ModelStage,MODEL_STAGE,DEVICE

# Modules
from src.object_detector.models.object_detector_factory import ObjectDetector

from src.binary_classifier.models.binary_classifier_selection_region_factory import BinaryClassifierSelectionRegion
from src.binary_classifier.models.binary_classifier_region_abnormal_factory import BinaryClassifierRegionAbnormal

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

        if MODEL_STAGE==ModelStage.CLASSIFIER.value:
            self.binary_classifier_selection_region = BinaryClassifierSelectionRegion().create_model()
            self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal().create_model()
    
        

    def forward(self,images: Tensor , object_detector_targets: Optional[List[Dict[str, Tensor]]] = None, selection_classifier_targets: Tensor=None,abnormal_classifier_targets: Tensor = None):
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
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,0,0
            # Stage(2) Binary Classifier
            object_detector_detected_classes=object_detector_detected_classes.to(DEVICE)
            selection_classifier_losses,_,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,_=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses
       
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
            