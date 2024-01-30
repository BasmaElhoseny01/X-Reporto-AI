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

from config import DEVICE
import sys

'''
image is grey scale 1*512*512 [Grey Scale 512*512]  
(with feature maps of size 16x16) of 2048 channels [2048*16*16] 
backbone: 
'''

class FrcnnObjectDetectorV1(nn.Module):
    def __init__(self):
        '''
        The constructor for Object Detector model
        Args:

        Returns:

        '''
        super().__init__()

        # No of Classes 29 Anatomical Region + Back Ground
        self.num_classes = 30

        # Loading pre-trained resnet50
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modifying first Conv Layer to take Grey Scale image instead of rgb
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


        # Removing Last 2 layers (avgpool,fc) for the Resnet to use as feature extractor without classification
        self.backbone=nn.Sequential(*list(resnet.children())[:-2])

        # # Defining the out_channel for the backbone = out for the last conv layer in Layer(4) (2048)
        self.backbone.out_channels=resnet.layer4[-1].conv3.out_channels

        # Anchor Aspect Ratios and Size since the input image size is 512 x 512, we choose the sizes accordingly
        # Suiting 29 Anatomical Region
        anchor_generator = AnchorGenerator(
            sizes=((20, 40, 60, 80, 100, 120, 140, 160, 180, 300),),
            aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 5.0, 8.0),),
        )

        # @Basma Elhoseny TODO Check RPN HEAD + RPN
        rpn_head = RPNHead(self.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

        self.rpn=RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=0.7, # minimum IoU between the anchor and the GT box so that they can be considered as positive during training of the RPN.
            bg_iou_thresh=0.3, # maximum IoU between the anchor and the GT box so that they can be considered as negative during training of the RPN.
            batch_size_per_image=256,
            positive_fraction=0.5, #  proportion of positive anchors in a mini-batch during training of the RPN
            pre_nms_top_n={"training": 2000, "testing": 1000},  # number of proposals to keep before applying NMS
            post_nms_top_n={"training": 2000, "testing": 1000}, # number of proposals to keep after applying NMS
            nms_thresh=0.7, # NMS threshold
            score_thresh=0.0,        
        )


        # @Basma Elhoseny TODO Check ROI
        # size of feature maps after roi pooling layer 8*8
        feature_map_output_size = 8

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=feature_map_output_size, sampling_ratio=2)

        resolution = roi_pooler.output_size[0]
        representation_size = 1024

        box_head = TwoMLPHead(self.backbone.out_channels * resolution**2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, self.num_classes)

        self.roi_heads=RoIHeads(
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,

            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,

            score_thresh=0.01,
            nms_thresh=0.0,
            detections_per_img=100,
        )

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
            (II) in eval mode:
                - losses (Dict[Tensor]). empty Dictionary {}
                - detections (List[Dict[str, Tensor]]), predictions for each input image
                    - boxes
                    - labels
                    - scores
        """
        # Features extracted from backbone feature map is 16*16 depth is 2048 [batch_size x 2048 x 16 x 16]
        features_maps=self.backbone(images)

        # Transform images and features from tensors to types that the rpn and roi_heads expect in the current PyTorch implementation.
        # Images have to be of class ImageList
        batch_size = images.shape[0] # batch_size
        image_sizes = images.shape[-2:] # 512*512 (input images dim)
        images = ImageList(images,image_sizes=[tuple(image_sizes) for _ in range(batch_size)])
        # Features have to be a dict where the str "0" maps to the features.
        features_maps = OrderedDict([("0", features_maps)])

        # Getting Proposals of RPN Bounding Boxes
        # In case of Training proposal_losses is Dictionary {"loss_objectness","loss_rpn_box_reg"} else it is None
        proposals, proposal_losses = self.rpn(images, features_maps, targets)

        detections, detector_losses  = self.roi_heads(features_maps, proposals, images.image_sizes, targets)

        # Getting Top 1 Region for each label
        detected_boxes=[]
        detected_labels=[]
        detected_features=[]

        # TODO REMOVE THAT
        detected_labels=[t['labels'] for t in targets]
        detected_boxes=[t['boxes'] for t in targets]
        detected_features = torch.randn(batch_size, 29, 1024).to(DEVICE)




        for pred in detections:
            # For each Image
            boxes,labels=self.get_top_k_boxes_for_labels(pred["boxes"], pred["labels"], pred["scores"], k=1)
            detected_boxes.append(boxes)
            detected_labels.append(labels)

            # TODO REMOVE THAT
            features= torch.randn(batch_size, 29, 1024)
            detected_features.append(features)
        # detected_features= torch.tensor([x.to(DEVICE) for x in detected_features]).to(DEVICE)
        


        # Losses for RPN Network and ROI Network
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        if self.training:
            return losses,None,detected_labels,detected_features
        else:
            return losses,detected_boxes,detected_labels,detected_features

    def get_top_k_boxes_for_labels(self, boxes, labels, scores, k=1):
        '''
        Function that returns the top k boxes for each label.

        inputs:
            boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
            labels: list of labels (Format [N] => N times label)
            scores: list of scores (Format [N] => N times score)
            k: number of boxes to return for each label
        outputs:
            listboxes: list of boxes maxlength 29 one box for each region
            labels: list of integers from 1 to 30 label for each box in listboxes
        '''
        # create a dict that stores the top k boxes for each label
        top_k_boxes_for_labels = {}
        # get the unique labels
        unique_labels = torch.unique(labels)
        # for each unique label
        for label in unique_labels:
            # get the indices of the boxes that have that label
            indices = torch.where(labels == label)[0]
            # get the scores of the boxes that have that label
            scores_for_label = scores[indices]
            # get the boxes that have that label
            boxes_for_label = boxes[indices]
            # sort the scores for that label in descending order
            sorted_scores_for_label, sorted_indices = torch.sort(scores_for_label, descending=True)
            # get the top k scores for that label
            top_k_scores_for_label = sorted_scores_for_label[:k]
            # get the top k boxes for that label
            top_k_boxes_for_label = boxes_for_label[sorted_indices[:k]]
            # store the top k boxes for that label
            top_k_boxes_for_labels[label] = top_k_boxes_for_label
            #convert boxes to list
        listboxes=[]
        for b in top_k_boxes_for_labels.values():
            b=b[0].tolist()
            listboxes.append(b)
        if len(unique_labels)!=0:
            return listboxes,unique_labels.tolist()
        return listboxes,[]



# model=FrcnnObjectDetectorV1()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device, non_blocking=True)


# from torchvision import transforms
# import numpy as np
# # Generate random data (simulating an RGB image with values between 0 and 255)
# random_image_data = np.random.randint(0, 256, size=(512, 512, 1), dtype=np.uint8)

# # Convert NumPy array to PyTorch tensor
# random_image_tensor = torch.from_numpy(random_image_data.transpose((2, 0, 1))).float()

# # Normalize the tensor (assuming you want to use it with a pre-trained model)
# normalize = transforms.Normalize(mean=[0.485], std=[0.229])
# normalized_tensor = normalize(random_image_tensor / 255.0)

# # Add an extra dimension to simulate batch size of 1
# input_image = normalized_tensor.unsqueeze(0)


# targets = []
# batch_size=1
# num_boxes=28

# for _ in range(batch_size):
#     # Generate random ground-truth boxes (x1, y1, x2, y2)
#     boxes = torch.rand((num_boxes, 4))
#     boxes[:, 2:] += boxes[:, :2]  # Ensure x1 < x2 and y1 < y2

#     # # Generate random ground-truth labels
#     # labels = torch.randint(low=1, high=29, size=(num_boxes,), dtype=torch.int64)
#     # labels = torch.unique(labels)  # Ensure unique class labels
#     labels = torch.arange(1, 30)[:num_boxes] 

#     # Create a dictionary for each batch element
#     target_dict = {"boxes": boxes, "labels": labels}

    
#     # Append the dictionary to the targets list
#     targets.append(target_dict)

# # Move all tensors in targets to GPU
# for target_dict in targets:
#     for key, value in target_dict.items():
#         if isinstance(value, torch.Tensor):
#             target_dict[key] = value.to(device)

# input_image=input_image.to(device)
# model.eval()
# print(model(input_image,targets))
# # Print or access the main layers
