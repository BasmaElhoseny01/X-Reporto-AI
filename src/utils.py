import torch
from torch import Tensor
from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib import patches

def boolean_to_indices(boolean_tensor: torch.Tensor) -> List[List[int]]:
    """
    Convert a 2D boolean tensor to a list of indices where the value is True.

    Parameters:
    - boolean_tensor (torch.Tensor): 2D boolean tensor.

    Returns:
    - 2D List representing the indices of True values.

    Example:
    >>> boolean_tensor = torch.tensor([[True, True, False, False, False, False, False, False, False, False, False, False, False],
    ...                                [False, False, False, False, False, False, True, False, False, False, False, False, False]])
    >>> indices_list = boolean_to_indices(boolean_tensor)
    >>> print(indices_list)
    [[1, 2], [7]]

    """
    # Get indices where the values are True
    true_indices = [[idx.item() + 1 for idx in torch.nonzero(row)] for row in boolean_tensor]

    return true_indices

def indices_to_boolean(indices_list: List[List[int]], num_columns: int=29) -> torch.Tensor:
    """
    Convert a list of indices to a 2D boolean tensor.

    Parameters:
    - indices_list (List[List[int]]): List of indices.
    - num_columns (int): Number of columns in the resulting boolean tensor.(Default =29 :D Anatomical Region)

    Returns:
    - torch.Tensor: 2D boolean tensor with True values at specified indices.

    Example:
    >>> example_indices = [[1, 2], [7]]
    >>> num_columns = 13
    >>> result_boolean_tensor = indices_to_boolean(example_indices, num_columns)
    >>> print(result_boolean_tensor)
    tensor([[True, True, False, False, False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, True, False, False, False, False, False, False]])
    """
    boolean_tensor = torch.zeros(len(indices_list), num_columns, dtype=torch.bool)

    for i, indices_row in enumerate(indices_list):
        for index in indices_row:
            if 0 < index <= num_columns:
                boolean_tensor[i, index - 1] = True

    return boolean_tensor


import torch
from typing import List

def select_boxes_with_boolean_mask(boxes_tensor: torch.Tensor, boolean_mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Select boxes from a 3D tensor based on a boolean mask.

    Parameters:
    - boxes_tensor (torch.Tensor): 3D tensor of bounding boxes.
    - boolean_mask (torch.Tensor): 2D boolean mask indicating which boxes to select.

    Returns:
    - List[torch.Tensor]: List of selected boxes.

    Example:
    >>> object_detector_boxes = torch.tensor([[[366.40, 271.72, 507.02, 512.00],
    ...                                        [405.54, 0.00, 475.85, 37.33],
    ...                                        [4, 0.00, 475.85, 37.33],
    ...                                        [5.54, 0.00, 475.85, 37.33],
    ...                                        [55.54, 0.00, 475.85, 37.33],
    ...                                        [500.54, 0.00, 475.85, 37.33],
    ...                                        [400.72, 0.00, 479.33, 36.11]],
    ...                                        [[1.40, 271.72, 507.02, 512.00],
    ...                                         [2.54, 0.00, 475.85, 37.33],
    ...                                         [4, 0.00, 475.85, 37.33],
    ...                                         [0.54, 0.00, 475.85, 37.33],
    ...                                         [5.54, 0.00, 475.85, 37.33],
    ...                                         [0.54, 0.00, 475.85, 37.33],
    ...                                         [0.72, 0.00, 479.33, 36.11]]
    ...                                         ], device='cuda:0')
    >>> boolean_mask = torch.tensor([[True, True, False, False, False, False, True],
    ...                               [False, False, False, False, False, False, True]])
    >>> selected_boxes = select_boxes_with_boolean_mask(object_detector_boxes, boolean_mask)
    >>> print(selected_boxes)
    [tensor([[366.4000, 271.7200, 507.0200, 512.0000],
             [405.5400,   0.0000, 475.8500,  37.3300],
             [400.7200,   0.0000, 479.3300,  36.1100]], device='cuda:0'),
     tensor([[400.7200,   0.0000, 479.3300,  36.1100]], device='cuda:0')]
    """
    selected_boxes = []

    for i, mask_row in enumerate(boolean_mask):
        selected_indices = torch.nonzero(mask_row).squeeze(1)
        selected_boxes.append(boxes_tensor[i, selected_indices])

    return selected_boxes



def get_top_k_boxes_for_labels(boxes, labels, scores, k=1):
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

# display the image with the bounding boxes
# pridected boxes are solid and the true boxes are dashed
def plot_image(img,labels, boxes,prdictedLabels,prdictedBoxes):
    '''
    Function that draws the BBoxes on the image.

    inputs:
        img: input-image as numpy.array (shape: [H, W, C])
        labels: list of integers from 1 to 30
        boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
        prdictedLabels: list of bool
        prdictedBoxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    '''
    cmap = plt.get_cmap("tab20b")
    height, width = img.shape[1:]
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    # Display the image
    ax.imshow(img[0])
    region_colors = ["b", "g", "r", "c", "m", "y"]
    for j in range(0,5):
        for i in range(j*6,j*6+5):
            if labels[i]:
                box = boxes[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    width,
                    height,
                    linewidth=1,  # Increase linewidth
                    # make the box color correspond to the label color
                    edgecolor=region_colors[((i-j*6)%5)],
                    # edgecolor="white",  # Set the box border color
                    facecolor="none",  # Set facecolor to none
                    linestyle="dashed",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
            if prdictedLabels[i]:
                box = prdictedBoxes[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    width,
                    height,
                    linewidth=1,  # Increase linewidth
                    # make the box color correspond to the label color
                    edgecolor=region_colors[(i-j*6)%5],
                    # edgecolor="white",  # Set the box border color
                    facecolor="none",  # Set facecolor to none
                    linestyle="solid",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
        plt.show()
        cmap = plt.get_cmap("tab20b")
        height, width = img.shape[1:]
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(16, 8))
        # Display the image
        ax.imshow(img[0])

# display the image with the bounding boxes (only true boxes or only predicted boxes)
def plot_single_image(img, boxes):
    '''
    Function that draws the BBoxes on the image.

    inputs:
        img: input-image as numpy.array (shape: [H, W, C])
        boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    '''
    cmap = plt.get_cmap("tab20b")
    height, width = img.shape[1:]
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    # Display the image
    ax.imshow(img[0])
    for i, box in enumerate(boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle(
            (box[0], box[1]),
            width,
            height,
            linewidth=1,  # Increase linewidth
            edgecolor="white",  # Set the box border color
            facecolor="none",  # Set facecolor to none
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()