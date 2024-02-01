import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional

# Modules for The Binary Classifier
from src.binary_classifier.models.binary_classifier_region_abnormal_v1 import BinaryClassifierRegionAbnormalV1

class BinaryClassifierRegionAbnormalWrapper(nn.Module):
    '''
    Wrapper class for a binary classifier that identifies abnormal regions.

    Parameters:
        - abnormal_binary_classifier (nn.Module): Binary classifier for abnormal region detection.

    Methods:
        - forward(object_detector_features: Tensor, object_detector_detected_classes: Tensor, abnormal_classifier_targets: Optional[Tensor] = None)
          Performs a forward pass through the binary classifier.
    '''

    def __init__(self, abnormal_binary_classifier):
        super().__init__()
        self.abnormal_binary_classifier = abnormal_binary_classifier

    def forward(self, object_detector_features: Tensor, object_detector_detected_classes:Tensor,abnormal_classifier_targets:Optional[Tensor] = None):
        '''
        Forward pass for the Binary Classifier for selecting Abnormal Regions

        Args:
            - region_features (Tensor): Tensor of shape [batch_size x 29 x 1024].
                Features for the regions detected by the object detector.
            - class_detected (Tensor): Boolean tensor of shape [batch_size x 29].
                Indicates if the object detector has detected the region/class or not.
            - region_is_abnormal (Optional[Tensor]): Boolean tensor of shape [batch_size x 29].
                Ground truth if a region is abnormal (True) or not (False). Default is None.
                If None, the model assumes inference mode without ground truth labels.

        returns:
            If in training mode:
            - loss(Tensor): Loss of the Binary Classifier. The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)`.
            - predicted_abnormal_regions: None
            If in Validation mode:
            - loss(Tensor): Loss of the Binary Classifier. The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)`.
            - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
              Indicating predicted abnormal regions.
            If in Inference mode:
            - loss: None
            - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
              Indicating predicted abnormal regions.
        '''
        # Modify Input/Output as the Required by submodule
        if self.training:
            classifier_losses=self.abnormal_binary_classifier(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            predicted_abnormal_regions=None
        else:
            classifier_losses,predicted_abnormal_regions=self.abnormal_binary_classifier(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)

        return classifier_losses,predicted_abnormal_regions

class BinaryClassifierRegionAbnormal():
    """
    Main class for creating and managing a binary classifier for abnormal region detection.

    Methods:
    - create_model() -> BinaryClassifierRegionAbnormalWrapper:
        Creates an instance of the binary classifier wrapper.

    Example:
    >>> binary_classifier = BinaryClassifierRegionAbnormal()
    >>> wrapper_model = binary_classifier.create_model()
    """
    def __init__(self):
        pass
    
    def create_model(self) -> BinaryClassifierRegionAbnormalWrapper:
        """
        Creates an instance of the binary classifier wrapper.

        Returns:
        - BinaryClassifierRegionAbnormalWrapper: An instance of the binary classifier wrapper.
        """
        BinaryClassifierAbnormal_module=BinaryClassifierRegionAbnormalV1()
        return BinaryClassifierRegionAbnormalWrapper(BinaryClassifierAbnormal_module)