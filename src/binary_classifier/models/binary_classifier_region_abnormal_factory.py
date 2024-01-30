import torch
from torch import Tensor
import torch.nn as nn
import copy
from src.binary_classifier.models.binary_classifier_region_abnormal_v1 import BinaryClassifierRegionAbnormalV1

class BinaryClassifierRegionAbnormalWrapper(nn.Module):
    def __init__(self, abnormal_binary_classifier):
        super().__init__()
        self.abnormal_binary_classifier = abnormal_binary_classifier

    def forward(self, object_detector_features: Tensor, object_detector_detected_classes:Tensor,abnormal_classifier_targets: Tensor):
        # Modify Input/Output as the Required by submodule
        if self.training:
                    classifier_losses=self.abnormal_binary_classifier(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
                    predicted_abnormal_regions=None
        else:
            pass

        return classifier_losses,predicted_abnormal_regions

class BinaryClassifierRegionAbnormal():
    def __init__(self):
        pass
    
    def create_model(self) -> BinaryClassifierRegionAbnormalWrapper:
        BinaryClassifierAbnormal_module=BinaryClassifierRegionAbnormalV1()
        return BinaryClassifierRegionAbnormalWrapper(BinaryClassifierAbnormal_module)

# model=BinaryClassifierRegionAbnormal().create_model()
# print(model)