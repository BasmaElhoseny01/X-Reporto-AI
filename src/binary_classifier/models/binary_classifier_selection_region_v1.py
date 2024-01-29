import torch
import torch.nn as nn
import torch.nn.functional as F
class BinaryClassifierSelectionRegionV1(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1):
        super(BinaryClassifierSelectionRegionV1, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.binary_classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_dim)
        )
        # we set pos_weight=2.2 to put 2.2 more weight on the loss of regions with sentences
        pos_weight = torch.tensor([2.2], device=self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    

    def forward(
        self,
        top_region_features,  # tensor of shape [batch_size x 29 x 1024]
        class_detected,  # boolean tensor of shape [batch_size x 29], indicates if the object detector has detected the region/class or not
        return_loss,  # boolean value that is True if we need the loss (necessary for training and evaluation)
        region_has_sentence  # boolean tensor of shape [batch_size x 29], indicates if a region has a sentence (True) or not (False) as the ground truth
    ):
        # logits of shape [batch_size x 29]
        logits = self.binary_classifier(top_region_features).squeeze(dim=-1)

        # the loss is needed for training and evaluation
        if return_loss:
        # only compute loss for logits that correspond to a class that was detected (class_detected=True as can be not existing in image)
            detected_logits = logits[class_detected]
            detected_region_has_sentence = region_has_sentence[class_detected]

            loss = self.loss_fn(detected_logits, detected_region_has_sentence.type(torch.float32))

        if self.training:
            return loss
        else:
            # if a logit > -1 (log2(0.5)=-1)
            selected_regions = logits > -1 

            # set to False all regions that were not detected by object detector
            # (since no detection -> no sentence generation possible)
            selected_regions[~class_detected] = False

            # selected_region_features are inputted into the decoder during evaluation and inference to generate the sentences
            # selected_region_features is of shape [num_regions_selected_in_batch, 1024]
            selected_region_features = top_region_features[selected_regions]

            # Val mode
            if return_loss:
                return loss, selected_regions, selected_region_features
            else:
                # Test mode
                return selected_regions, selected_region_features

        
    