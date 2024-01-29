from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn

from config import ModelStage,MODEL_STAGE
from src.object_detector.models.object_detector_factory import ObjectDetector

class XReportoV1(nn.Module):
    def __init__(self):
        '''
        The constructor for X-Reporto model
        Args:

        Returns:

        '''
        super().__init__()

        self.object_detector = ObjectDetector().create_model()
        

    def forward(self,images: Tensor , object_detector_targets: Optional[List[Dict[str, Tensor]]] = None):
        '''
        Args:
            - images (Tensor): images_tensor [batch_size x 1 x 512 x 512] (grey-scaled images) [Normalized 0-1]
            - object_detector_targets (List[Dict[str, Tensor]]) [Optional]: List of dict for target of each batch element :
                - boxes (FloatTensor[N, 4]): ground-truth boxes [x1,y1,x2,y2] format , N=bounding boxes detected
                - labels (Int64Tensor[N]): ground-truth labels for each box , N=bounding boxes detected
            - classifier_targets (List[]) [Optional]: List of dict for target of each batch element :
            - language_model__targets (List[]) [Optional]: List of dict for target of each batch element :
        '''

        if self.training:
            # Training

            # Stage(1) Object Detector
            object_detector_losses,object_detector_detections= self.object_detector(images=images, targets=object_detector_targets)

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,object_detector_detections
            
            # Stage(2) Binary Classifier

        else:
            # Evaluation

            # Stage(1) Object Detector
            object_detector_detections= self.object_detector(images=images, targets=object_detector_targets)

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_detections

            # Stage(2) Binary Classifier
            