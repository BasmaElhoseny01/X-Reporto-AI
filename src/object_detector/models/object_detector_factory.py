from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn

import sys
# Version For Object Detector
from src.object_detector.models.frccn_object_detector_v1 import FrcnnObjectDetectorV1

class ObjectDetectorWrapper(nn.Module):
    '''
    Wrapper class for an object detector module.

    Parameters:
        - object_detector: The object detector module to be wrapped.

    Methods:
        - forward(images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
           Performs a forward pass through the object detector.

    '''
    def __init__(self, object_detector):
        super().__init__()
        self.object_detector = object_detector

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        '''
        Performs a forward pass through the object detector.

        Args:
            - images (Tensor):  Tensor of shape [batch_size, 1, 512, 512] ([Grey Scaled],Normalized(0-1)) 
                The tensor containing the input images to be processed by the object detector.
            - targets (Optional[List[Dict[str, Tensor]]]): List of dictionaries containing Bounding Boxes targets.
                If None, the model assumes inference mode without ground truth labels.
                Each dictionary in the list should have the following keys:
                - 'boxes' (Tensor): Tensor of bounding box coordinates with shape [num_boxes, 4].
                - 'labels' (Tensor): Tensor of class labels with shape [num_boxes,].

        Returns:
            If in training mode:
                - object_detector_losses (Dict): Dictionary containing object detector losses with keys:
                    - 'loss_objectness' (Tensor): Objectness loss.
                    - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                    - 'loss_classifier' (Tensor): Classifier loss.
                    - 'loss_box_reg' (Tensor): Box regression loss.
                - object_detector_boxes: None
                - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                    Indicates if the object detector has detected the region/class or not.
                - object_detector_features (Tensor): Tensor of shape [batch_size x 29 x 1024].
                    Region Features detected by the object detector.
            
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
                - object_detector_features (Tensor): Tensor of shape [batch_size x 29 x 1024].
                    Region Features detected by the object detector.

            If in Inference mode:           
                - object_detector_losses (Dict): Empty Dictionary
                - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                    Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                    Indicates if the object detector has detected the region/class or not.
                - object_detector_features (Tensor): Tensor of shape [batch_size x 29 x 1024].
                    Region Features detected by the object detector.
        '''

        # Modify Input/Output as the Required by submodule
        if self.training:
            object_detector_losses, object_detector_predictions =self.object_detector(images,targets)
            object_detector_features=object_detector_predictions['features']
            object_detector_detected_classes=object_detector_predictions['class_detected'] # [batch size x 29]            
            object_detector_boxes=None


        else:
            object_detector_losses, object_detector_predictions =self.object_detector(images,targets)
            object_detector_features=object_detector_predictions['features']
            object_detector_detected_classes=object_detector_predictions['class_detected'] # [batch size x 29]
            object_detector_boxes=object_detector_predictions['detections'] ["top_region_boxes"]# [batch size x 29 x 4]
            
        return object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features

class ObjectDetector():
    """
    Main class for creating and managing anObject Detector.

    Methods:
    - create_model() -> ObjectDetectorWrapper:
        Create and return the Object Detector model.

    Example:
    >>> object_detector = ObjectDetector()
    >>> wrapper_model = object_detector.create_model()
    """
    def __init__(self):
        pass
    
    def create_model(self) -> ObjectDetectorWrapper:
        '''
        Create and return the Object Detector model.

        Returns:
            ObjectDetectorWrapper: The wrapped Object Detector model.
        '''
        # Add Required Object Detector Module
        
        object_detector_module=FrcnnObjectDetectorV1(features=True)
        
        return ObjectDetectorWrapper(object_detector_module)