import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional, List, Dict

# Modules for The Binary Classifier
from src.binary_classifier.models.binary_classifier_selection_region_v1 import BinaryClassifierSelectionRegionV1

class BinaryClassifierSelectionRegionWrapper(nn.Module):
    '''
    Wrapper class for a binary classifier that identifies regions for selection.

    Parameters:
        - selection_binary_classifier (nn.Module): Binary classifier for selecting regions.

    Methods:
        - forward(self, object_detector_features: Tensor, object_detector_detected_classes:Tensor,selection_classifier_targets: Tensor=None)
          Performs a forward pass through the binary classifier.
    '''
    def __init__(self, selection_binary_classifier):
        super().__init__()
        self.selection_binary_classifier = selection_binary_classifier

    def forward(self, object_detector_features: Tensor, object_detector_detected_classes:Tensor,selection_classifier_targets: Tensor=None):
        '''
        Forward pass for the Binary Classifier for selecting regions with sentences.

        Args:
            - region_features (Tensor): Tensor of shape [batch_size x 29 x 1024].
                Features for the regions detected by the object detector.
            - class_detected (Tensor): Boolean tensor of shape [batch_size x 29].
                Indicates if the object detector has detected the region/class or not.
            - region_has_sentence (Optional[Tensor]): Boolean tensor of shape [batch_size x 29].
                Ground truth if a region has a sentence (True) or not (False). Default is None.
                If None, the model assumes inference mode without ground truth labels.
        Returns:
            If in training mode:
                - loss(Tensor): Loss of the Binary Classifier. The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)`.
                - selected_regions: None
                - selected_region_features:None

            If in Validation mode:
                - loss(Tensor): Loss of the Binary Classifier. The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)`.
                - selected_regions(Tensor): Tensor of shape [batch_size x 29] 
                  Indicating selected regions
                - selected_region_features(Tensor): Boolean Tensor of Shape [num_regions_selected_in_batch,4]
                  Representing features of selected regions
                
            If in Inference mode:
                - loss: None
                - selected_regions(Tensor): Tensor of shape [batch_size x 29] 
                  Indicating selected regions
                  # TODO @Basma Elhoseny Check that when integrating LM
                - selected_region_features(Tensor): Boolean Tensor of Shape [num_regions_selected_in_batch,4]
                  Representing features of selected regions
        '''
        # Modify Input/Output as the Required by submodule
        if self.training:
                    classifier_losses=self.selection_binary_classifier(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
                    selected_regions=None
                    selected_region_features=None
        else:
            classifier_losses,selected_regions, selected_region_features=self.selection_binary_classifier(object_detector_features,object_detector_detected_classes,selection_classifier_targets)

        return classifier_losses,selected_regions,selected_region_features
class BinaryClassifierSelectionRegion():
    '''
    Main class for creating and managing a binary classifier for region selection.

    Methods:
    - create_model() -> BinaryClassifierSelectionRegionWrapper:
        Creates an instance of the binary classifier wrapper for region selection.

    Example:
    >>> region_selection_classifier = BinaryClassifierSelectionRegion()
    >>> wrapper_model = region_selection_classifier.create_model()
    '''
    def __init__(self):
        pass
    
    def create_model(self) -> BinaryClassifierSelectionRegionWrapper:
        '''
        Creates an instance of the binary classifier wrapper for region selection.

        Returns:
        - BinaryClassifierSelectionRegionWrapper: An instance of the binary classifier wrapper for region selection.
        '''
        BinaryClassifierSelectionRegion_module=BinaryClassifierSelectionRegionV1()
        return BinaryClassifierSelectionRegionWrapper(BinaryClassifierSelectionRegion_module)
