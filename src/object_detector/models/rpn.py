from typing import Dict, List, Optional, Tuple
from torch import nn

from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    concat_box_prediction_layers,
)
from torchvision.models.detection.image_list import ImageList
from torch import Tensor


class Rpn(RegionProposalNetwork):
    def __init__(
            self,
            head: nn.Module,
            anchor_generator: AnchorGenerator,
            # Faster-RCNN Training
            fg_iou_thresh: float=0.7,
            bg_iou_thresh: float=0.3,
            batch_size_per_image: int=256,
            positive_fraction: float=0.5,
            # Faster-RCNN Inference
            pre_nms_top_n: Dict[str, int]=dict(training=2000, testing=1000),
            post_nms_top_n: Dict[str, int]=dict(training=2000, testing=1000),
            nms_thresh: float=0.7,
            score_thresh: float = 0.0,
        ):
        super().__init__(
            anchor_generator,
            head,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            pre_nms_top_n,
            post_nms_top_n,
            nms_thresh,
            score_thresh,
        )
            

    def forward(self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        its same implementation of RegionProposalNetwork.forward but make it return the losses
        in case of evaluation and give targets as input
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image.(Optional)
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if targets is not None:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

