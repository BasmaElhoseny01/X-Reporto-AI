import torch
from torch import Tensor
from typing import Dict, List, Union
import shutil
import numpy as np
import random
import os
import subprocess

import matplotlib.pyplot as plt
from matplotlib import patches
import io
# import tensorflow as tf

from config import *

from sklearn import metrics

def ROC_AUC(y_true,y_scores):
    """
    Compute ROC curve and AUC (Area Under the Curve).

    Parameters:
        y_true (array-like): True binary labels. 
        y_scores (array-like): Target scores, can either be probability estimates of the positive class or confidence values.

    Returns:
        fpr (array-like): False Positive Rate (FPR).
        tpr (array-like): True Positive Rate (TPR).
        auc (float): Area Under the ROC Curve (AUC).
        optimal_threshold (float): Optimal threshold based on Youden's J statistic.
    """
    # ROC Curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)

    # AUC
    auc = metrics.auc(fpr, tpr)

    # Optimal Threshold
    # Compute Youden's J statistic
    j_statistic = tpr - fpr

    # Find the index of the threshold that maximizes J statistic
    optimal_threshold_index = np.argmax(j_statistic)

    # Get the optimal threshold
    optimal_threshold = thresholds[optimal_threshold_index]

    return fpr, tpr,auc,optimal_threshold


def plot_to_image():
    """
    Convert a matplotlib plot to a TensorFlow tensor image.

    Returns:
        image (numpy.ndarray): NumPy array representing the image.
    """
    # Create a buffer to store the plot
    buf = io.BytesIO()
    
    # Save the plot to the buffer
    plt.savefig(buf, format='png')
    
    # Close the plot to avoid memory leaks
    plt.close()
    
    # Convert the buffer to a numpy array
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Expand the dimensions of the image tensor
    image = tf.expand_dims(image, 0)

    image = tf.squeeze(image, axis=0)  # Remove the first dimension (batch size)

    # Convert the TensorFlow EagerTensor image to a NumPy array
    image = image.numpy()

    image=image[:, :, :3]  # Take only the first three channels (RGB)

    return image

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

def plot_image(img: np.ndarray,img_idx:int, labels: List[int], boxes: List[List[float]], predicted_labels: List[bool], predicted_boxes: List[List[float]],selected_region:bool=False,target_regions:List[bool]=[]):
    """
    Function that draws the BBoxes on the image.

    Args:
        - img (np.ndarray): Input image as numpy.array (shape: [H, W, C]).
        - labels (List[int]): List of integers representing class labels (1 to 30).
        - boxes (List[List[float]]): List of ground truth bounding boxes.
            Format: [N, 4] => N times [xmin, ymin, xmax, ymax].
        - predicted_labels (List[bool]): List of boolean values representing predicted labels.
        - predicted_boxes (List[List[float]]): List of predicted bounding boxes.
            Format: [N, 4] => N times [xmin, ymin, xmax, ymax].

    Returns:
        None

    Notes:
        - Predicted boxes are drawn in solid lines.
        - Ground truth boxes are drawn in dashed lines.
    """
    cmap = plt.get_cmap("tab20b")
    height, width = img.shape[1:]
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    # Display the image
    ax.imshow(img[0])
    region_colors = [
        "blue", "green", "red", "cyan", "magenta", "yellow", 
        "orange", "purple", "lime", "pink", "teal", "lavender", 
        "brown", "beige", "maroon", "mint", "coral", "olive", 
        "navy", "grey", "black", "white", "indigo", "turquoise", 
        "salmon", "khaki", "gold", "violet", "tan"
    ]

    images_list=[]
    for j in range(0,5):
        count=0
        for i in range(j*6,j*6+5):
            condition= False
            idx= -1
            if i in labels:
                if selected_region:
                    if target_regions[count]:
                        idx=i
                        condition=True
                        count+=1
                else:
                    idx=i
                    condition=True

            if condition and idx !=-1:
                box = boxes[idx]
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    width,
                    height,
                    linewidth=1,  # Increase linewidth
                    # make the box color correspond to the label color
                    edgecolor=region_colors[(i-j*6-1)%29],
                    # edgecolor="white",  # Set the box border color
                    facecolor="none",  # Set facecolor to none
                    linestyle="dashed",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
            if predicted_labels[i]:
                box = predicted_boxes[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    width,
                    height,
                    linewidth=1,  # Increase linewidth
                    # make the box color correspond to the label color
                    edgecolor=region_colors[(i-j*6)%29],
                    # edgecolor="white",  # Set the box border color
                    facecolor="none",  # Set facecolor to none
                    linestyle="solid",
                )
                # Add the patch to the Axes
                ax.add_patch(rect)
                
        # images_list.append(ax)
        # convert ax to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images_list.append(img)
        # plt.show()
        

        
    return images_list
    # save_path="assets/"+str(RUN)+"/image_"+str(img_idx+1)+"_region"+str(j)+".png"
        # plt.savefig(save_path)
        # plt.close()
                
        # plt.show()
        # cmap = plt.get_cmap("tab20b")
        # height, width = img.shape[1:]

        # # Create figure and axes
        # fig, ax = plt.subplots(1, figsize=(16, 8))

        # # Display the image
        # ax.imshow(img[0])

def plot_single_image(img: np.ndarray, boxes: List[List[float]],grayscale:bool=False ,save_path: str = None):
    """
    Function that draws bounding boxes on the image.

    Args:
        img (np.ndarray): Input image as numpy array (shape: [H, W, C]).
        boxes (List[List[float]]): List of bounding boxes.
            Format: [N, 4] => N times [xmin, ymin, xmax, ymax].
        save_path (str, optional): Path to save the image. If None, the image will be displayed instead. Defaults to None.
        grayscale (bool, optional): Whether to display the image in grayscale. Defaults to False.

    Returns:
        None

    Notes:
        - If save_path is provided, the image will be saved with bounding boxes.
        - If save_path is None, the image will be displayed with bounding boxes.
        - If grayscale is True, the image will be displayed in grayscale.
    """
    # Maximum number of colors
    max_colors = 30
    
    # Generate distinct colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(max_colors)]

    height, width = img.shape[0:2]
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))

    # Display the image
    if grayscale:
        ax.imshow(np.squeeze(img), cmap='gray')
    else:
        ax.imshow(img)
    
    for i, box in enumerate(boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        color = colors[i % max_colors]  # Select color from the generated list
        rect = patches.Rectangle(
            (box[0], box[1]),
            width,
            height,
            linewidth=1,  # Increase linewidth
            edgecolor=color,  # Set the box border color
            facecolor="none",  # Set facecolor to none
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
    
    if save_path:
        plt.savefig(save_path)
        print("Image Saved Sucessfully at ",save_path)
    else:
        plt.show()

# Example usage:
# plot_single_image(image_array, bounding_boxes, save_path="output_image.png")
def save_model(model,name):
    '''
    Save the X-Reporto model to a file.

    Args:
        model(nn): model to be saved
        name (str): Name of the model file.
    '''
    torch.save(model.state_dict(), "models/" + str(RUN) + '/' + name + ".pth")

    if SAVE_TO_DRIVE:
        try:
          # Source and destination file paths
          source_path = f"models/{RUN}/{name}.pth"
          destination_path = f"/content/drive/MyDrive/models/{RUN}/{name}.pth"

          destination_dir = os.path.dirname(destination_path)
          if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
          # Save to Drive
          subprocess.run(["cp", source_path, destination_path])
          logging.info(f"Model Saved Successfully to drive")
            
        except Exception as e:
            logging.info(f"Failed to save model to Drive. Reason: {e}")

def load_model(model,name):
    '''
    Load the X-Reporto model from a file.

    Args:
        model(nn): model to be loaded
        name (str): Name of the model file.
    '''
    # if LOAD_FROM_DRIVE:
    #     try:
    #         # Load from Drive
    #         subprocess.run(["cp", f"/content/drive/MyDrive/models/" + str(RUN) + '/' + name + ".pth", f"models/" + str(RUN) + '/' + name + ".pth"])
    #     except Exception as e:
    #         print(f"Failed to load model from Drive. Reason: {e}")
    model.load_state_dict(torch.load("models/" + str(RUN) + '/' + name + ".pth"))

def plot_heatmap():
    pass

def cuda_memory_info(title=""):
    print("==========================================================================================================")
    print("Memory after ["+title+"]")
    num_cuda_devices = torch.cuda.device_count()
    for i in range(num_cuda_devices):
        device = torch.cuda.get_device_properties(i)
        total_memory = device.total_memory / 1024**3  # Total memory available on the device
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # Memory currently in use by tensors
        reserved_memory = torch.cuda.memory_reserved(i) / 1024**3  # Total memory reserved by PyTorch
        remaining_memory_current = total_memory - allocated_memory  # Remaining memory that is currently available for allocation
        remaining_memory_potential = total_memory - reserved_memory  # Remaining memory that can potentially be allocated
        print(f"Device {i}:")
        print(f"  Name: {device.name}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Allocated Memory: {allocated_memory:.2f} GB")
        print(f"  Reserved Memory: {reserved_memory:.2f} GB")
        print(f"  Remaining Memory (Current): {remaining_memory_current:.2f} GB")
        print(f"  Remaining Memory (Potential): {remaining_memory_potential:.2f} GB")
    print("==========================================================================================================")


def save_checkpoint(epoch:int,batch_index:int,optimizer_state:Dict,scheduler_state_dict,model_state:Dict,best_loss:float,epoch_loss:float):
    checkpoint={
    "model_state":model_state, #
    "scheduler_state_dict":scheduler_state_dict, #
    "optimizer_state":optimizer_state,#
    "best_loss":best_loss, #
    "epoch":epoch,#
    "epoch_loss":epoch_loss,#
    "batch_index":batch_index,#
    # "config":get_config()
    }

    # # Save Checkpoint by time 
    # # Get the current date and time
    # current_datetime = datetime.datetime.now()
    # # Format the date and time to be part of the filename
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # # Create the filename with the formatted datetime
    # name = f"ckpt_{formatted_datetime}"

    checkpoint_path='check_points/'+str(RUN)+'/checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    logging.info('Saved Check point at ' + checkpoint_path)

    if SAVE_TO_DRIVE:
      try:
          # Destination directory
          destination_dir = f"/content/drive/MyDrive/check_points/{RUN}/"

          # Check if the destination directory exists, create it if not
          if not os.path.exists(destination_dir):
              os.makedirs(destination_dir)

          # Copy the checkpoint file to the destination directory
          subprocess.run(["cp", checkpoint_path, f"{destination_dir}checkpoint.pth"])
          
          logging.info("Checkpoint saved successfully to Drive.")
      except Exception as e:
          logging.info(f"Failed to save checkpoint to Drive. Reason: {e}")
            

def load_checkpoint(run):
    checkpoint_path='check_points/'+str(run)+'/checkpoint.pth'
    logging.info('Loading Check point at' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def seed_worker(worker_id):
    """To preserve reproducibility for the randomly shuffled train loader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def empty_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove all files and subdirectories within the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")
        logging.info(f"Folder '{folder_path}' emptied successfully.")
    else:
        logging.info(f"Folder '{folder_path}' does not exist.")