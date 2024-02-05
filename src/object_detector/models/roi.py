from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads,fastrcnn_loss
from torchvision.ops import boxes as box_ops
from torch import nn
class Roi(RoIHeads):
    def __init__(self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        # Faster R-CNN inference
        score_thresh=0.01,
        nms_thresh=0.0,
        detections_per_img=100,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        # return features is train in all model or not if train individual model
        features=False,
        feature_size=8,

        ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            # Faster R-CNN training
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            # Faster R-CNN inference
            score_thresh,
            nms_thresh,
            detections_per_img,
            # Mask
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )
        self.features=features
        # set kernel_size = feature_map_output_size, such that we average over the whole feature maps
        self.avg_pool = nn.AvgPool2d(kernel_size=feature_size)
        self.dim_reduction = nn.Linear(2048, 1024)   

    def get_top_k_boxes_for_labels(
        self,
        box_features,
        box_regression,
        class_logits,
        proposals,
        image_shapes
    ):
        """
        Method returns an output dict containing different values depending on if:
            - the object detector is used in isolation (i.e. self.return_feature_vectors == False) or as part of the full model (i.e. self.return_feature_vectors == True)
            - we are in train or eval mode

        The possibilities are:

        (1) object detector is used in isolation + eval mode:
            -> output dict contains the keys "detections" and "class_detected":

            - "detections" maps to another dict with the keys "top_region_boxes" and "top_scores":
                - "top_region_boxes" maps to a tensor of shape [batch_size, 29, 4] of the detected boxes with the highest score (i.e. top-1 score) per class
                - "top_scores" maps to a tensor of shape [batch_size, 29] of the corresponding highest scores for the boxes

            - "class_detected" maps to a boolean tensor of shape [batch_size, 29] that has a True value for a class if that class had the highest score (out of all classes)
            for at least 1 proposed box. If a class has a False value, this means that for all hundreds of proposed boxes coming from the RPN for a single image,
            this class did not have the highest score (and thus was not predicted/detected as the class) for one of them. We use the boolean tensor of "class_detected"
            to mask out the boxes for these False/not-detected classes in "detections"

        (2) object detector is used with full model + train mode:
            -> output dict contains the keys "top_region_features" and "class_detected":

            - "top_region_features" maps to a tensor of shape [batch_size, 29, 2048] of the region features with the highest score (i.e. top-1 score) per class
            - "class_detected" same as above. Needed to mask out the region features for classes that were not detected later on in the full model

        (3) object detector is used with full model + eval mode:
            -> output dict contains the keys "detections", "top_region_features", "class_detected":
            -> all keys same as above
        """
        # apply softmax on background class as well
        # (such that if the background class has a high score, all other classes will have a low score)
        pred_scores = F.softmax(class_logits, -1)

        # remove score of the background class
        pred_scores = pred_scores[:, 1:]

        # get number of proposals/boxes per image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        num_images = len(boxes_per_image)

        # split pred_scores (which is a tensor with scores for all RoIs of all images in the batch)
        # into the tuple pred_scores_per_img (where 1 pred_score tensor has scores for all RoIs of 1 image)
        pred_scores_per_img = torch.split(pred_scores, boxes_per_image, dim=0)

        # if we train/evaluate the full model, we need the top region/box features
        if self.features:
            # split region_features the same way as pred_scores
            region_features_per_img = torch.split(box_features, boxes_per_image, dim=0)
        else:
            region_features_per_img = [None] * num_images  # dummy list such that we can still zip everything up

        # if we evaluate the object detector, we need the detections
        if not self.training:
            pred_region_boxes = self.box_coder.decode(box_regression, proposals)
            pred_region_boxes_per_img = torch.split(pred_region_boxes, boxes_per_image, dim=0)
        else:
            pred_region_boxes_per_img = [None] * num_images  # dummy list such that we can still zip everything up

        output = {}
        output["class_detected"] = []  # list collects the bool arrays of shape [29] that specify if a class was detected (True) for each image
        output["top_region_features"] = []  # list collects the tensors of shape [29 x 2048] of the top region features for each image

        # list top_region_boxes collects the tensors of shape [29 x 4] of the top region boxes for each image
        # list top_scores collects the tensors of shape [29] of the corresponding top scores for each image
        output["detections"] = {
            "top_region_boxes": [],
            "top_scores": []
        }
        # loop on each image and its corresponding predicted scores, region features, region boxes and image shape
        for pred_scores, region_features, pred_region_boxes, image_shape in zip(
            pred_scores_per_img, region_features_per_img, pred_region_boxes_per_img, image_shapes
        ):
            # get predicted class for each region/box
            pred_classes = pred_scores.argmax(dim=1)
            # create a mask that is 1 at the predicted class index for every box and 0 otherwise
            mask_pred_classes = torch.nn.functional.one_hot(pred_classes, num_classes=29).to(pred_scores.device)
            # multiply the predicted scores with the mask to get the predicted score for the predicted class
            pred_top_score = pred_scores * mask_pred_classes
            # get the scores and row indices of the box/region features with the top-1 score for each class (dim=0 goes by class)
            top_scores, indices_with_top_scores = torch.max(pred_top_score, dim=0)
            num_predictions_per_class = torch.sum(mask_pred_classes, dim=0)
            class_detected = (num_predictions_per_class > 0)
            output["class_detected"].append(class_detected)
            if self.features:
                # get the top region features
                top_region_features = region_features[indices_with_top_scores]
                # append the top region features to the list
                output["top_region_features"].append(top_region_features)
            if not self.training:
                # clip boxes so that they lie inside an image of size "img_shape"
                pred_region_boxes = box_ops.clip_boxes_to_image(pred_region_boxes, image_shape)
                pred_region_boxes = pred_region_boxes[:, 1:]
                top_region_boxes = pred_region_boxes[indices_with_top_scores, torch.arange(start=0, end=29, dtype=torch.int64, device=indices_with_top_scores.device)]
                output["detections"]["top_region_boxes"].append(top_region_boxes)
                output["detections"]["top_scores"].append(top_scores)
             # convert lists into batched tensors
        output["class_detected"] = torch.stack(output["class_detected"], dim=0)  # of shape [batch_size x 29]

        if self.features:
            output["top_region_features"] = torch.stack(output["top_region_features"], dim=0)  # of shape [batch_size x 29 x 2048]

        if not self.training:
            output["detections"]["top_region_boxes"] = torch.stack(output["detections"]["top_region_boxes"], dim=0)  # of shape [batch_size x 29 x 4]
            output["detections"]["top_scores"] = torch.stack(output["detections"]["top_scores"], dim=0)  # of shape [batch_size x 29]

        return output

    def forward(self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if targets is not None:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None  

        box_roi_pool_feature = self.box_roi_pool(features, proposals, image_shapes)
        box_head_feature = self.box_head(box_roi_pool_feature)
        class_logits, box_regression = self.box_predictor(box_head_feature)

        detection_losses = {}

        if labels and regression_targets:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detection_losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }

        roi_heads_output = {}
        roi_heads_output["detector_losses"] = detection_losses

        if self.features or not self.training:
            # average over the spatial dimensions, i.e. transform roi pooling features maps from [num_proposals, 2048, 8, 8] to [num_proposals, 2048, 1, 1]
            box_features = self.avg_pool(box_roi_pool_feature)

            # remove all dims of size 1
            box_features = torch.squeeze(box_features)

            output = self.get_top_k_boxes_for_labels(box_features, box_regression, class_logits, proposals, image_shapes)

            roi_heads_output["class_detected"] = output["class_detected"]

            if self.features:
                # transform top_region_features from [batch_size x 29 x 2048] to [batch_size x 29 x 1024]
                roi_heads_output["top_region_features"] = self.dim_reduction(output["top_region_features"])

            if not self.training:
                roi_heads_output["detections"] = output["detections"]

        return roi_heads_output            

