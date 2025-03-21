from typing import Optional, List, Dict
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
import torchvision

from torchvision.models.detection.image_list import ImageList

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator,RPNHead,RegionProposalNetwork

from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from src.object_detector.models.feature_extraction import FeatureNetwork
from src.object_detector.models.roi import Roi

from src.object_detector.models.rpn import Rpn
from src.object_detector.models.feature_extraction import FeatureNetwork
import sys

from config import TRAIN_ROI, TRAIN_RPN

'''
image is grey scale 1*512*512 [Grey Scale 512*512]  
(with feature maps of size 16x16) of 2048 channels [2048*16*16] 
backbone: 
'''

class FrcnnObjectDetectorV1(nn.Module):
    def __init__(self,features=True):
        '''
        The constructor for Object Detector model
        Args:

        Returns:

        '''
        super().__init__()
        self.features=features
        # No of Classes 29 Anatomical Region + Back Ground
        self.num_classes = 30

        self.backbone = FeatureNetwork("resnet50")
        # Anchor Aspect Ratios and Size since the input image size is 512 x 512, we choose the sizes accordingly
        # Suiting 29 Anatomical Region
        anchor_generator = AnchorGenerator(
            sizes=((20, 40, 60, 80, 100, 120, 140, 160, 180, 300),),
            aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 5.0, 8.0),),
        )
        rpn_head = RPNHead(self.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
        self.rpn=Rpn(rpn_head,anchor_generator)
        # size of feature maps after roi pooling layer 8*8
        feature_map_output_size = 8

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=feature_map_output_size, sampling_ratio=2)

        resolution = roi_pooler.output_size[0]
        representation_size = 1024

        box_head = TwoMLPHead(self.backbone.out_channels * resolution**2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, self.num_classes)

        self.roi_heads=Roi(
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            features=self.features,
            feature_size=feature_map_output_size,
        )
        # freez RPN and Backbone layers if train_roi
        if TRAIN_ROI:
            for param in self.rpn.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _check_targets(self, targets):
        """
        Check if
            - there are targets for training
            - all bboxes are of correct type and shape
            - there are no degenerate bboxes (where e.g. x1 > x2)
        """
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")

        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            if not isinstance(boxes, torch.Tensor):
                torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

            torch._assert(
                len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
            )

            # x1 should always be < x2 and y1 should always be < y2
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width." f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    
    def _transform_inputs_for_rpn_and_roi(self, images, features):
        """
        Tranform images and features from tensors to types that the rpn and roi_heads expect in the current PyTorch implementation.

        Concretely, images have to be of class ImageList, which is a custom PyTorch class.
        Features have to be a dict where the str "0" maps to the features.

        Args:
            images (Tensor)
            features (Tensor): of shape [batch_size, 2048, 16, 16]

        Returns:
            images (ImageList)
            features (Dict[str, Tensor])
        """
        images = ImageList(images)
        features = OrderedDict([("0", features)])

        return images, features

    def forward(self,images: Tensor ,targets: Optional[List[Dict[str, Tensor]]] = None):
        """
        Args:
            - images (Tensor): images_tensor [batch_size x 1 x 512 x 512] (grey-scaled images) [Normalized 0-1]
            - targets(List[Dict[str, Tensor]]) List of dict for target of each batch element :
                - boxes (FloatTensor[N, 4]): ground-truth boxes [x1,y1,x2,y2] format , N=bounding boxes detected
                - labels (Int64Tensor[N]): ground-truth labels for each box , N=bounding boxes detected

        Returns:
            (I) in train mode:
                - losses (Dict[Tensor]), which contains the 4 object detector losses
                    - loss_objectness
                    - loss_rpn_box_reg
                    - loss_classifier
                    - loss_box_reg
                - features (Tensor): [batch_size x 29 x 1024] features of each bounding box
            (II) in eval mode:
                 case: features=False

            losses = {
                "loss_objectness",
                "loss_rpn_box_reg",
                "loss_classifier",
                "loss_box_reg",
            }
            outputs={
                "class_detected":[batcg_size * 29 ] true or false,
                "detections":{
                    "top_region_boxes",
                    "top_scores",
                }
            }
            case: features=True
            
            losses = {
                "loss_objectness",
                "loss_rpn_box_reg",
                "loss_classifier",
                "loss_box_reg",
            }
            outputs={
                "class_detected":[batcg_size * 29 ] true or false,
                "detections":{
                    "top_region_boxes",
                    "top_scores",
                },
                "top_region_features"
            }
        """
        # Features extracted from backbone feature map is 16*16 depth is 2048 [batch_size x 2048 x 16 x 16]
        features=self.backbone(images)
        # Transform images and features from tensors to types that the rpn and roi_heads expect in the current PyTorch implementation.
        # Images have to be of class ImageList
        batch_size = images.shape[0] # batch_size
        image_sizes = images.shape[-2:] # 512*512 (input images dim)
        images = ImageList(images,image_sizes=[tuple(image_sizes) for _ in range(batch_size)])
        # Features have to be a dict where the str "0" maps to the features.
        features = OrderedDict([("0", features)])
        # Getting Proposals of RPN Bounding Boxes
        # In case of Training proposal_losses is Dictionary {"loss_objectness","loss_rpn_box_reg"} else it is None
        proposals, proposal_losses = self.rpn(images, features, targets)


        if TRAIN_RPN:
            # Turn Off Object ROI output 
            detector_losses={}
            outputs = {}
            outputs["features"]=None
            outputs["class_detected"] = None
            outputs["class_detected"] = None
            outputs["detections"]={}
            outputs["detections"]["top_region_boxes"]=None
            outputs["class_detected"] = None
            outputs["features"]=None
        else:
            detections  = self.roi_heads(features, proposals, images.image_sizes, targets)
            detector_losses =detections["detector_losses"]

            outputs = {}
            if not self.training:
                outputs["class_detected"] = detections["class_detected"]
                outputs["detections"]=detections["detections"]
            if self.features:
                outputs["class_detected"] = detections["class_detected"]
                outputs["features"]=detections["top_region_features"]

        if TRAIN_ROI:
            proposal_losses = {}
        # Losses for RPN Network and ROI Network
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        
        if self.training and self.features:
            return losses,outputs
        
        if self.training:
            return losses
        
        else:
            return losses,outputs

'''
model=FrcnnObjectDetectorV1()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device, non_blocking=True)


from torchvision import transforms
import numpy as np
# Generate random data (simulating an RGB image with values between 0 and 255)
random_image_data = np.random.randint(0, 256, size=(512, 512, 1), dtype=np.uint8)

# Convert NumPy array to PyTorch tensor
random_image_tensor = torch.from_numpy(random_image_data.transpose((2, 0, 1))).float()

# Normalize the tensor (assuming you want to use it with a pre-trained model)
normalize = transforms.Normalize(mean=[0.485], std=[0.229])
normalized_tensor = normalize(random_image_tensor / 255.0)

# Add an extra dimension to simulate batch size of 1
input_image = normalized_tensor.unsqueeze(0)


targets = []
batch_size=1
num_boxes=28

for _ in range(batch_size):
    # Generate random ground-truth boxes (x1, y1, x2, y2)
    boxes = torch.rand((num_boxes, 4))
    boxes[:, 2:] += boxes[:, :2]  # Ensure x1 < x2 and y1 < y2

    # # Generate random ground-truth labels
    # labels = torch.randint(low=1, high=29, size=(num_boxes,), dtype=torch.int64)
    # labels = torch.unique(labels)  # Ensure unique class labels
    labels = torch.arange(1, 30)[:num_boxes] 

    # Create a dictionary for each batch element
    target_dict = {"boxes": boxes, "labels": labels}

    
    # Append the dictionary to the targets list
    targets.append(target_dict)

# Move all tensors in targets to GPU
for target_dict in targets:
    for key, value in target_dict.items():
        if isinstance(value, torch.Tensor):
            target_dict[key] = value.to(device)

input_image=input_image.to(device)
model.eval()
print(model(input_image,targets))
# Print or access the main layers

'''