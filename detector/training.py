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
from data_loader import F_RCNNDataset, Augmentation
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


from matplotlib import patches
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

class Trainer:
    def __init__(self,debug=False,training_csv_path='datasets/overfitting.csv',validation_csv_path='datasets/overfitting.csv',
                 model_path='model.pth',load_model=False,batch_size=1,epochs=1, learning_rate=0.0001):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.debug = debug
        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # create model object detector
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        
        num_classes = 30 # 29 class (abnormal) + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        self.bestloss=100000

        self.model.to(self.device)

        # load model if specified
        if load_model:
            self.model.load_state_dict(torch.load(model_path))

        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.learning_rate)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

        # create loss function for classification and regression
        # caluclated in the model and returned as a dictionary
        # self.criterion_cls = nn.CrossEntropyLoss()
        # self.criterion_reg = nn.MSELoss()

        # create dataset
        self.dataset_train = F_RCNNDataset(dataset_path= training_csv_path)
        self.dataset_val = F_RCNNDataset(dataset_path= validation_csv_path)
        
        # create data loader
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.data_loader_val = DataLoader(dataset=self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4)
        

    def train(self):
        # train the model
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.lr_scheduler.step()
            
        self.evaluate_one_epoch()

            # save the model
            # torch.save(self.model.state_dict(), 'model.pth')
    def train_one_epoch(self):
        self.model.train()
        # train_loss_list=[]
        for batch_idx, (images, targets) in enumerate(self.data_loader_train):
            # add new dimension to images after batch size
            images = images.unsqueeze(1)
            images = list(image.to(self.device) for image in images)
            targetdata=[]
            # correct => targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image
            # what i get => targets return as dic with keys: boxes, labels Only
            for i in range(len(images)):
                newdic={}
                newdic['boxes']=targets['boxes'][i]
                newdic['labels']=targets['labels'][i]
                targetdata.append(newdic)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            loss_dict = self.model(images, targetdata)    

            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            # train_loss_list.append(loss_value)

            # save the best model
            # comment for now ##########################################
            # if(loss_value<self.bestloss):
            #     self.bestloss=loss_value
            #     torch.save(self.model.state_dict(), 'bestmodel.pth')

            losses.backward()
            self.optimizer.step()

            # print statistics
            if self.debug:
                print(f'Batch {batch_idx + 1}/{len(self.data_loader_train)} Loss: {losses.item():.4f}')
                print(loss_value)
            break
    

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


    def evaluate_one_epoch(self):
        self.model.eval()
        # val_loss_list=[]
        for batch_idx, (images, targets) in enumerate(self.data_loader_val):
            images = images.unsqueeze(1)
            images = list(image.to(self.device) for image in images)
            targetdata=[]
            # correct => targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image
            # what i get => targets return as dic with keys: boxes, labels Only
            for i in range(len(images)):
                newdic={}
                newdic['boxes']=targets['boxes'][i]
                newdic['labels']=targets['labels'][i]
                targetdata.append(newdic)
            # forward
            with torch.no_grad():
                # loss_dict = self.model(images, targetdata)
                # # During testing, it returns list[BoxList] contains additional fields
                # # like `scores`, `labels` and `mask` (for Mask R-CNN models).
                # losses = sum(loss for loss in loss_dict.values())
                # loss_value = losses.item()
                prediction = self.model(images)[0]
                # apply NMS to prediction on boxes with score > 0.5
                prediction = self.model(images)[0]
                keep = torchvision.ops.nms(prediction["boxes"], prediction["scores"], 0.3)
                prediction["boxes"] = prediction["boxes"][keep]
                prediction["scores"] = prediction["scores"][keep]
                prediction["labels"] = prediction["labels"][keep]
                # get the scores, boxes and labels
                scores = prediction["scores"].tolist()
                boxes = prediction["boxes"].tolist()
                labels = prediction["labels"].tolist()
                targetBoxes=targets["boxes"].tolist()
                # print(type(targetBoxes),len(targetBoxes),targetBoxes)
                # print(type(boxes),len(boxes),boxes)
                # compare the labels with targetdata labels 

                # self.showImg(prediction,labels,images[0])
                print(len(targetBoxes[0]))
                print(len(boxes))
                plot_image(images[0], targetBoxes[0])
                plot_image(images[0], boxes)
                # val_loss_list.append(loss_value)
    
                # print statistics
                if self.debug:
                    # print labels , scores , boxes
                    print("labels : ",labels)
                    print("scores : ",scores)
                    print("boxes : ",boxes)
                break

def plot_image(img, boxes):
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
if __name__ == '__main__':
    trainer = Trainer(debug=True)
    trainer.train()