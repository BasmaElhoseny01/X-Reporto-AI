import torch
import matplotlib.pyplot as plt
from matplotlib import patches

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
        prdictedLabels: list of integers from 1 to 30
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
        for i in range(j*6+1,j*6+7):
            if i in labels:
                index = labels.index(i)
                box = boxes[index]
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    width,
                    height,
                    linewidth=1,  # Increase linewidth
                    # make the box color correspond to the label color
                    edgecolor=region_colors[((i-j*6-1)%7)-1],
                    # edgecolor="white",  # Set the box border color
                    facecolor="none",  # Set facecolor to none
                    linestyle="dashed",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
            if i in prdictedLabels:
                index = prdictedLabels.index(i)
                box = prdictedBoxes[index]
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    width,
                    height,
                    linewidth=1,  # Increase linewidth
                    # make the box color correspond to the label color
                    edgecolor=region_colors[(i-j*6-1%7)-1],
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