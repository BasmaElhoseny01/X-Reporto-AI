from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import copy

from src.object_detector_copy.models.frccn_object_detector_v1 import FrcnnObjectDetectorV1
from src.object_detector_copy.models.object_detector_paper.object_detector import ObjectDetectorOriginal


class ObjectDetectorWrapper(nn.Module):
    def __init__(self, object_detector):
        super().__init__()
        self.object_detector = object_detector

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        # Modify Input/Output as the Required by submodule
        if self.training:
            object_detector_losses, object_detector_features, object_detector_labels=self.object_detector(images,targets)
            object_detector_detected_classes=copy.deepcopy(object_detector_labels)
            if(isinstance(self.object_detector, FrcnnObjectDetectorV1)):
                # from indices to true/false
                object_detector_detected_classes = [torch.zeros(self.object_detector.num_classes-1,dtype=torch.bool) for _ in object_detector_labels]
                for i in range(len(object_detector_detected_classes)):
                    object_detector_detected_classes[i][object_detector_labels[i] - 1] = True
                object_detector_detected_classes=torch.stack(object_detector_detected_classes)
            object_detector_boxes=None
        else:
            object_detector_losses,object_detector_boxes,object_detector_features,object_detector_detected_classes=self.object_detector(images,targets)
            object_detector_losses=None
            if(isinstance(self.object_detector, ObjectDetectorOriginal)):
                object_detector_boxes=object_detector_boxes['top_region_boxes']
                object_detector_detected_classes = [[idx.item() + 1 for idx in torch.nonzero(row)] for row in object_detector_detected_classes]
        return object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features

class ObjectDetector():
    def __init__(self):
        pass
    
    def create_model(self) -> ObjectDetectorWrapper:

        # Add Required Object Detector Module
        
        # object_detector_module=FrcnnObjectDetectorV1()
        object_detector_module=ObjectDetectorOriginal(return_feature_vectors=True)
        
        return ObjectDetectorWrapper(object_detector_module)

# model=ObjectDetector().create_model()
# print(model)
# model.train()
# model(10,20)
