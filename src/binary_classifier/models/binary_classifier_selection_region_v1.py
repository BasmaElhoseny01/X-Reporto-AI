import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional

from config import DEVICE,REGION_SELECTION_CLASSIFIER_POS_WEIGHT
class BinaryClassifierSelectionRegionV1(nn.Module):
    """
    Binary classifier for selecting regions with sentences.

    Parameters:
    - input_dim (int): Dimensionality of the input features. Default is 1024.
    - output_dim (int): Dimensionality of the output logits. Default is 1.
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

        # Setting pos_weight=2.2 to put 2.2 more weight on the loss of regions with sentences
        pos_weight = torch.tensor([REGION_SELECTION_CLASSIFIER_POS_WEIGHT], device=DEVICE)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    

    def forward(self,region_features,class_detected, region_has_sentence=None):
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
                - selected_region_features(Tensor): Boolean Tensor of shape [num_regions_selected_in_batch,4]
                  Representing features of selected regions

        '''
        # logits of shape [batch_size x 29]
        logits = self.binary_classifier(region_features).squeeze(dim=-1)

        loss=None
        if region_has_sentence is not None:
            # only compute loss for logits that correspond to a class that was detected
            detected_logits = logits[class_detected]
            detected_region_has_sentence = region_has_sentence[class_detected]
            loss = self.loss_fn(detected_logits, detected_region_has_sentence.type(torch.float32))
       
        if self.training:
            return loss
        else: # Validation (or inference) mode
            # if a logit > -1 (log2(0.5)=-1), then it means class/region has boolean value True and is has a sentence on it 
            selected_regions = logits > -1 

            # set to False all regions that were not detected by object detector, (since no detection -> no sentence generation possible)
            selected_regions[~class_detected] = False

            # selected_region_features are inputted into the decoder during evaluation and inference to generate the sentences
            selected_region_features = region_features[selected_regions] # [num_regions_selected_in_batch, 1024]


            return loss,selected_regions, selected_region_features

        
    