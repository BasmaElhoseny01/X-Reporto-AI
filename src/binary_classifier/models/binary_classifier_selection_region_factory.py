from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn
import copy
from src.binary_classifier.models.binary_classifier_selection_region_v1 import BinaryClassifierSelectionRegionV1
from config import DEVICE
import numpy as np

class BinaryClassifierSelectionRegionWrapper(nn.Module):
    def __init__(self, selection_binary_classifier):
        super().__init__()
        self.selection_binary_classifier = selection_binary_classifier

    def forward(self, object_detector_features: Tensor, object_detector_detected_classes:Tensor,selection_classifier_targets: Tensor=None):
        # Modify Input/Output as the Required by submodule
        if self.training:
                    classifier_losses=self.selection_binary_classifier(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
                    selected_regions=None
                    selected_region_features=None
        else:
            object_detector_labels=copy.deepcopy(object_detector_detected_classes)
            object_detector_detected_classes = [torch.zeros(29,dtype=torch.bool) for _ in object_detector_labels]
            for i in range(len(object_detector_detected_classes)):
                object_detector_detected_classes[i][np.array(object_detector_labels[i]) - 1] = True
            object_detector_detected_classes=torch.stack(object_detector_detected_classes).to(DEVICE)
            
            selected_regions, selected_region_features=self.selection_binary_classifier(object_detector_features,object_detector_detected_classes)
            classifier_losses=None
            selected_regions = [[idx.item() + 1 for idx in torch.nonzero(row)] for row in selected_regions]
        return classifier_losses,selected_regions,selected_region_features
class BinaryClassifierSelectionRegion():
    def __init__(self):
        pass
    
    def create_model(self) -> BinaryClassifierSelectionRegionWrapper:
        BinaryClassifierSelectionRegion_module=BinaryClassifierSelectionRegionV1()
        return BinaryClassifierSelectionRegionWrapper(BinaryClassifierSelectionRegion_module)
