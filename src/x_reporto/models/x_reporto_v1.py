from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn

from config import ModelStage,MODEL_STAGE,DEVICE

# from src.object_detector.models.object_detector_factory import ObjectDetector
from src.object_detector_copy.models.object_detector_factory import ObjectDetector

from src.binary_classifier.models.binary_classifier_selection_region_factory import BinaryClassifierSelectionRegion
from src.binary_classifier.models.binary_classifier_region_abnormal_factory import BinaryClassifierRegionAbnormal
import sys
class XReportoV1(nn.Module):
    def __init__(self):
        '''
        The constructor for X-Reporto model
        Args:

        Returns:

        '''
        super().__init__()
        self.num_classes=30

        self.object_detector = ObjectDetector().create_model()


        if MODEL_STAGE==ModelStage.CLASSIFIER.value:
            self.binary_classifier_selection_region = BinaryClassifierSelectionRegion().create_model()
            self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal().create_model()
    
        

    def forward(self,images: Tensor , object_detector_targets: Optional[List[Dict[str, Tensor]]] = None, selection_classifier_targets: Optional[List[Dict[str, Tensor]]] = None):
        '''
        Args:
            - images (Tensor): images_tensor [batch_size x 1 x 512 x 512] (grey-scaled images) [Normalized 0-1]
            - object_detector_targets (List[Dict[str, Tensor]]) [Optional]: List of dict for target of each batch element :
                - boxes (FloatTensor[N, 4]): ground-truth boxes [x1,y1,x2,y2] format , N=bounding boxes detected
                - labels (Int64Tensor[N]): ground-truth labels for each box , N=bounding boxes detected
            - selection_classifier_targets (List[]) [Optional]: List of dict for target of each batch element :
                -bbox_phrase_exists
            -  
        
            - language_model__targets (List[]) [Optional]: List of dict for target of each batch element :
        '''
        if self.training:
            # Training
            print("1")
            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_labels,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,object_detector_boxes,object_detector_labels,object_detector_features
            print("2")
            
            # Stage(2) Binary Classifier
            # Boolean array that is True for the classes that were detected
            object_detector_detected_classes = [torch.zeros(self.num_classes-1,dtype=torch.bool) for _ in object_detector_labels]


            for i in range(len(object_detector_detected_classes)):
                object_detector_detected_classes[i][object_detector_labels[i] - 1] = True
            object_detector_detected_classes=torch.stack(object_detector_detected_classes).to(DEVICE)


            selection_classifier_losses=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,True,selection_classifier_targets)
            print(selection_classifier_losses)
            sys.exit()

        else:
            # Evaluation

            # Stage(1) Object Detector
            object_detector_detections= self.object_detector(images=images, targets=object_detector_targets)

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_detections

            # Stage(2) Binary Classifier
            