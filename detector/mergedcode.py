import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pandas as pd
from PIL import Image
import tqdm
from data_loader import F_RCNNDataset, Augmentation
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


from matplotlib import patches
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


import torch.nn as nn


class Trainer(nn.Module):
    def __init__(self,traindata,validdata ,debug=False,training_csv_path='datasets/train-200.csv',validation_csv_path='datasets/train-200.csv',
                 model_path='model.pth',load_model=False,batch_size=1,epochs=20, learning_rate=0.0001):

        super().__init__()
        print(torch.cuda.is_available()) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("self.device : ",self.device)
        self.debug = debug
        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # create model object detector
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)


        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None,num_classes=30)
        num_classes = 30 # 29 class (abnormal) + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)       
        


        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=30)
        # num_classes = 30 # 29 class (abnormal) + background
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # # Create the custom classifier
        # classifier = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        # # Initialize the classifier weights with He initialization
        # nn.init.kaiming_uniform_(classifier.cls_score.weight, nonlinearity='relu')
        # nn.init.constant_(classifier.cls_score.bias, 0)
        # # Replace the existing classifier with the new one
        # self.model.roi_heads.box_predictor = classifier
        
        
        # self.model.rpn.anchor_generator= torchvision.models.detection.anchor_utils.AnchorGenerator(
        #     sizes=((20, 40, 60, 80, 100, 120, 140, 160, 180, 300),),
        #     aspect_ratios=((0.2, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.1, 2.6, 3.0, 5.0, 8.0),),
        # )


        self.bestloss=100000

        self.model.to(self.device)

        # load model if specified
        if load_model:
            self.model.load_state_dict(torch.load(model_path))

        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.learning_rate)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        # create loss function for classification and regression
        # caluclated in the model and returned as a dictionary
        # self.criterion_cls = nn.CrossEntropyLoss()
        # self.criterion_reg = nn.MSELoss()

        # create dataset
        # self.dataset_train = F_RCNNDataset(dataset_path= training_csv_path)
        # self.dataset_val = F_RCNNDataset(dataset_path= validation_csv_path)
        
        # # create data loader
        # self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, shuffle=False, num_workers=4)
        # self.data_loader_val = DataLoader(dataset=self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.data_loader_train=traindata
        self.data_loader_val=validdata
    
    def train(self):
        # train the model
        self.model.train()
        for epoch in range(self.epochs):
            # train_loss_list=[]
            # for batch_idx, (images, targets) in enumerate(self.data_loader_train):
            for batch_idx, data in enumerate(self.data_loader_train):
                # print("data : ",data)
                images = data['images']
                targets = data['targets']

                # add new dimension to images after batch size
                # images = images.unsqueeze(1)
                # print("images",images)
                # print("targets : ",targets)
                # images = list(image.to(self.device) for image in images)
                # targetdata=[]
                # correct => targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image
                # what i get => targets return as dic with keys: boxes, labels Only
                # for i in range(len(images)):
                #     newdic={}
                #     newdic['boxes']=targets['boxes'][i].to(self.device)
                #     newdic['labels']=targets['labels'][i].to(self.device)
                #     targetdata.append(newdic)

                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # print("epoch : ",batch_idx,"   targets = ",targets)
                # plot_image(images[0], targets[0]["boxes"])
                loss_dict = self.model(images, targets)   
                # print("loss_dict",loss_dict) 

                losses = sum(loss for loss in loss_dict.values())
                # print("losses",losses) 


                loss_value = losses.item()
                # train_loss_list.append(loss_value)

                # save the best model
                # comment for now ##########################################
                # if(loss_value<self.bestloss):
                #     self.bestloss=loss_value
                #     torch.save(self.model.state_dict(), 'bestmodel.pth')


                # BackWard
                losses.backward()
                
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update
                self.optimizer.step()
        

                if self.debug :
                    print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} Loss: {loss_value:.4f}')
            
            # self.lr_scheduler.step()
            # self.evaluate_one_epoch()

            # save the model
            # torch.save(self.model.state_dict(), 'model.pth')
        
    def showImg(self,prediction,labels,img):
        # conver image to unit8
        img=img.unit8()
        box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                            labels=labels,
                                            colors="red",
                                            width=4, font_size=30)
        im = to_pil_image(box.detach())
        im.show()
    import matplotlib.patches as patches


    def get_top_k_boxes_for_labels(self, boxes, labels, scores, k=1):
        '''
        Function that returns the top k boxes for each label.

        inputs:
            boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
            labels: list of labels (Format [N] => N times label)
            scores: list of scores (Format [N] => N times score)
            k: number of boxes to return for each label
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


            listboxes=[]
            for b in top_k_boxes_for_labels.values():
                # i need it to be 1d list instead of tensor([[394.2277,   0.5975, 512.0000,  51.7557]])
                b=b[0].tolist()
                listboxes.append(b)
            print("top_k_scores_for_label: ",top_k_scores_for_label)
        return listboxes,unique_labels.tolist()
    def evaluate_one_epoch(self):
        self.model.eval()
        # val_loss_list=[]
        for batch_idx, data in enumerate(self.data_loader_val):
            # images = images.unsqueeze(1)
            # images = list(image.to(self.device) for image in images)
            # targetdata=[]
            images = data['images']
            targets = data['targets']
            # correct => targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image
            # what i get => targets return as dic with keys: boxes, labels Only
            # for i in range(len(images)):
            #     newdic={}
            #     newdic['boxes']=targets['boxes'][i].to(self.device)
            #     newdic['labels']=targets['labels'][i].to(self.device)
            #     targetdata.append(newdic)
            # forward
            with torch.no_grad():
                # loss_dict = self.model(images, targetdata)
                # # During testing, it returns list[BoxList] contains additional fields
                # # like `scores`, `labels` and `mask` (for Mask R-CNN models).
                # losses = sum(loss for loss in loss_dict.values())
                # loss_value = losses.item()
                # prediction = self.model(images)[0]
                # apply NMS to prediction on boxes with score > 0.5
                prediction = self.model(images)[0]

                print("pridicted  : ",prediction)

                # keep = torchvision.ops.nms(prediction["boxes"], prediction["scores"], 0.3)
                print("#"*50)
                boxes,labels=self.get_top_k_boxes_for_labels(prediction["boxes"], prediction["labels"], prediction["scores"], k=1)
                # print("boxes : ",boxes)
                print("labels : ",labels)
                print("actual labels : ",targets[0]["labels"].tolist())
                # keep = prediction["scores"] >0.5
                
                # prediction["boxes"] = prediction["boxes"][keep]
                # prediction["scores"] = prediction["scores"][keep]
                # prediction["labels"] = prediction["labels"][keep]
                # # get the scores, boxes and labels
                # scores = prediction["scores"].tolist()
                # boxes = prediction["boxes"].tolist()
                # labels = prediction["labels"].tolist()
                targetBoxes=targets[0]["boxes"]
                # print(type(targetBoxes),len(targetBoxes),targetBoxes)
                # print(type(boxes),len(boxes),boxes)
                # compare the labels with targetdata labels 

                # self.showImg(prediction,labels,images[0])

                # actual result
                # plot_image(images[0], targetBoxes)
                plot_image(images[0], targets[0]["labels"].tolist(),targets[0]["boxes"].tolist(),labels,boxes)
                # val_loss_list.append(loss_value)
    
                # print statistics
                if self.debug ==False:
                    # print labels , scores , boxes
                    # print("labels : ",labels)
                    # print("scores : ",scores)
                    print("boxes : ",boxes)
                break

def plot_image(img,labels, boxes,prdictedLabels,prdictedBoxes):
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


    region_colors = ["b", "g", "r", "c", "m", "y"]

    for i in range(1,7):
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
                edgecolor=region_colors[(i%7)-1],
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
                edgecolor=region_colors[(i%7)-1],
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
    for i in range(7,13):
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
                edgecolor=region_colors[((i-6)%7) -1 ],
                # edgecolor="white",  # Set the box border color
                facecolor="none",  # Set facecolor to none
                # make it dashed
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
                edgecolor=region_colors[((i-6)%7)-1],
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
    for i in range(13,19):
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
                # make the box color correspond to the label color and make i
                edgecolor=region_colors[((i-12)%7)-1],
                # edgecolor="white",  # Set the box border color
                facecolor="none",  # Set facecolor to none
                # make it dashed
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
                edgecolor=region_colors[((i-12)%7)-1],
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
    for i in range(19,25):
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
                # make the box color correspond to the label color and make i
                edgecolor=region_colors[((i-18)%7)-1],
                # edgecolor="white",  # Set the box border color
                facecolor="none",  # Set facecolor to none
                # make it dashed
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
                edgecolor=region_colors[((i-18)%7)-1],
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
    plt.show()
    for i in range(25,31):
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
                # make the box color correspond to the label color and make i
                edgecolor=region_colors[((i-24)%7)-1],
                # edgecolor="white",  # Set the box border color
                facecolor="none",  # Set facecolor to none
                # make it dashed
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
                edgecolor=region_colors[((i-24)%7)-1],
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
    plt.show()



from ast import literal_eval
import os
import random
from typing import Dict, List
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch import Tensor
import torch
from custom_image_dataset_object_detector import CustomImageDataset
from torch.utils.data import DataLoader

PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 1.0
IMAGE_INPUT_SIZE = 512
SEED = 41
BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 8

def get_datasets_as_dfs():
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels"]

    # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}

    datasets_as_dfs = {dataset: os.path.join(os.getcwd()+"/datasets", dataset) + ".csv" for dataset in ["train", "valid"]}
    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}
    for dataset in datasets_as_dfs.values():
        # drop rows with NaN values
        dataset.dropna(how='all')
    # update all mimic_image_file_paths to point to the correct location
    for dataset in datasets_as_dfs.values():
        dataset["mimic_image_file_path"] = dataset["mimic_image_file_path"].apply(lambda x: os.path.join(os.getcwd(), x))
    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])
    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)
    print("new_num_samples_train : ",new_num_samples_train)
    print("new_num_samples_val : ",new_num_samples_val)
    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    return datasets_as_dfs

def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms

def collate_fn(batch: List[Dict[str, Tensor]]):
    # each dict in batch (which is a list) is for a single image and has the keys "image", "boxes", "labels"

    # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
    # otherwise, whole training loop will stop (even if only 1 image fails to open)
    
    
    batch = list(filter(lambda x: x is not None, batch))
    image_shape = batch[0]["image"].size()
    # allocate an empty images_batch tensor that will store all images of the batch
    images_batch = torch.empty(size=(len(batch), *image_shape))

    for i, sample in enumerate(batch):
        # remove image tensors from batch and store them in dedicated images_batch tensor
        images_batch[i] = sample.pop("image")

    # since batch (which is a list) now only contains dicts with keys "boxes" and "labels", rename it as targets
    targets = batch

    # create a new batch variable to store images_batch and targets
    batch_new = {}
    batch_new["images"] = images_batch
    batch_new["targets"] = targets

    return batch_new

def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def get_data_loaders(train_dataset, val_dataset):
    
    g = torch.Generator()
    g.manual_seed(SEED)

    # make shuffle=False for train_loader, since we want to try overfitting
    # train_loader = DataLoader(train_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    # val_loader = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

def main():
    datasets_as_dfs = get_datasets_as_dfs()
    train_transforms = get_transforms("val")
    val_transforms = get_transforms("val")

    train_dataset = CustomImageDataset(datasets_as_dfs["train"], train_transforms)
    val_dataset = CustomImageDataset(datasets_as_dfs["valid"], val_transforms)

    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)
    trainer = Trainer(traindata=train_loader, validdata=val_loader,debug=True)
    trainer.train()
    trainer.evaluate_one_epoch()
if __name__ == '__main__':
    main()