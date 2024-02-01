import torch
import torch.nn as nn
class BinaryClassifierRegionAbnormalV1(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.binary_classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_dim)
        )
        # we set pos_weight=6.0 to put 6.0 more weight on the loss of abnormal regions
        pos_weight = torch.tensor([6.0], device=self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(
        self,
        top_region_features,  # tensor of shape [batch_size x 29 x 1024]
        class_detected,  # boolean tensor of shape [batch_size x 29], indicates if the object detector has detected the region/class or not
        region_is_abnormal=None  # ground truth boolean tensor of shape [batch_size x 29], indicates if a region is abnormal (True) or not (False)
    ):
        # logits of shape [batch_size x 29]
        logits = self.binary_classifier(top_region_features).squeeze(dim=-1)

        loss = None
        if region_is_abnormal is not None:
            # only compute loss for logits that correspond to a class that was detected
            detected_logits = logits[class_detected]
            detected_region_is_abnormal = region_is_abnormal[class_detected]
            loss = self.loss_fn(detected_logits, detected_region_is_abnormal.type(torch.float32))

        if self.training:
            return loss
        else: # only val mode
            # if a logit > -1, then it means that class/region has boolean value True and is considered abnormal
            predicted_abnormal_regions = logits > -1
            # regions that were not detected will be filtered out later (via class_detected) when computing recall, precision etc.
            return loss, predicted_abnormal_regions


        