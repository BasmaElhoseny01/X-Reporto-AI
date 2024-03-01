import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional


from config import DEVICE,ABNORMAL_CLASSIFIER_POS_WEIGHT
class BinaryClassifierRegionAbnormalV1(nn.Module):
    """
    Binary classifier for detecting abnormal regions.

    Args:
        input_dim (int): Number of input features for the classifier.
        output_dim (int): Number of output dimensions (default is 1 for binary classification).
    """
    def __init__(self, input_dim=1024, output_dim=1):
        super().__init__()

        # Binary classifier network
        self.binary_classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_dim)
        )

        # Setting pos_weight=6.0 to put 6.0 more weight on the loss of abnormal regions
        pos_weight = torch.tensor([ABNORMAL_CLASSIFIER_POS_WEIGHT], device=DEVICE)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self,region_features:Tensor ,class_detected:Tensor ,region_is_abnormal:Optional[Tensor] = None):
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

        Returns:
            If in training mode:
            - loss(Tensor): Loss of the Binary Classifier. The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)`.
            If in Validation mode:
            - loss(Tensor): Loss of the Binary Classifier. The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward0>)`.
            - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
              Indicating predicted abnormal regions.
            If in Inference mode:
            - loss: None
            - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
              Indicating predicted abnormal regions.
        '''
        # logits of shape [batch_size x 29 x 1024]
        logits = self.binary_classifier(region_features).squeeze(dim=-1) # (batch_size x 29 x 1024) ->

        loss=None
        if region_is_abnormal is not None:
            # only compute loss for logits that correspond to a class that was detected
            detected_logits = logits[class_detected]
            detected_region_is_abnormal = region_is_abnormal[class_detected]
            loss = self.loss_fn(detected_logits, detected_region_is_abnormal.type(torch.float32))
        

        if self.training:
            return loss
        else: # Validation (or inference) mode
            # if a logits > -1, then it means that class/region has boolean value True and is considered abnormal
            predicted_abnormal_regions = logits > -1

            # regions that were not detected will be filtered out later (via class_detected) when computing recall, precision etc.
            return loss, predicted_abnormal_regions


        