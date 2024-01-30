import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from ..data_loader.custom_dataset import CustomDataset
from matplotlib import patches
import numpy as np
import sys
from src.object_detector.models.object_detector_factory import ObjectDetector
# constants
EPOCHS=50
LEARNING_RATE=0.0001
BATCH_SIZE=1
SCHEDULAR_STEP_SIZE=1
SCHEDULAR_GAMMA=0.9999999999
DEBUG=True

class Object_detector_trainer:
    
    def __init__(self,training_csv_path='datasets/train.csv',validation_csv_path='datasets/train.csv',
                 model=None):
        '''
        inputs:
            training_csv_path: string => the path to the training csv file
            validation_csv_path: string => the path to the validation csv file
            model: the object detector model
        '''
        # connect to gpu if available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if model==None:
            # load the model from bestmodel.path
            self.model=ObjectDetector().create_model()
            state_dict=torch.load('bestmodel.pth')
            self.model.load_state_dict(state_dict)

        else:
            self.model = model
        self.model.to(self.device)

        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= LEARNING_RATE)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)

        # create dataset
        self.dataset_train = CustomDataset(dataset_path= training_csv_path, transform_type='val')
        self.dataset_val = CustomDataset(dataset_path= validation_csv_path, transform_type='val')
        
        # create data loader
        # TODO @Ahmed Hosny suffle Training Loaders
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        self.data_loader_val = DataLoader(dataset=self.dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # initialize the best loss to a large value
        self.bestloss=10000000
        self.evalbestloss=10000000
    
    def train(self,rpn_only=False):
        '''
        Function to train the object detector on training dataset
        '''
        # make model in trainning mode
        self.model.train()
        for epoch in range(EPOCHS):
            for batch_idx, (images, targets) in enumerate(self.data_loader_train):
                # add new dimension to images after batch size
                images = torch.stack([image.to(self.device) for image in images])
                # convert targets from dic with keys: boxes, labels Only to list[Dict[str, Tensor]]
                targetdata=[]
                for i in range(len(images)):
                    newdic={}
                    newdic['boxes']=targets['boxes'][i].to(self.device)
                    newdic['labels']=targets['labels'][i].to(self.device)
                    targetdata.append(newdic)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss_dict = self.model(images, targetdata)   
                losses = sum(loss for loss in loss_dict.values())
                # sum rpn losses
                if rpn_only:
                    losses = loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']
                loss_value = losses.item()

                # save the best model
                if(loss_value<self.bestloss):
                    self.bestloss=loss_value
                    torch.save(self.model.state_dict(), 'bestmodel.pth')

                # BackWard
                losses.backward()

                # Update
                self.optimizer.step()
        
                if DEBUG :
                    print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} Loss: {loss_value:.4f}')
                    break
            self.lr_scheduler.step()

    def evaluate(self):
        '''
        Function to evaluate the object detector on evaluation dataset
        '''

        # make the model in evaluation mode
        # self.model.eval()
        # in evaluation mode the model will not return losses just predicted boxes so we cant calculate loss
        # until we implement customize roi and rpn to calculate loss
        self.model.train()
        for batch_idx, (images, targets) in enumerate(self.data_loader_val):
            # add new dimension to images after batch size
            # images = images.unsqueeze(1)
            images = torch.stack([image.to(self.device) for image in images])
            # convert targets from dic with keys: boxes, labels Only to list[Dict[str, Tensor]]
            targetdata=[]
            for i in range(len(images)):
                newdic={}
                newdic['boxes']=targets['boxes'][i].to(self.device)
                newdic['labels']=targets['labels'][i].to(self.device)
                targetdata.append(newdic)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            loss_dict = self.model(images, targetdata)   
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            # save the best model
            if(loss_value<self.evalbestloss):
                self.evalbestloss=loss_value
                torch.save(self.model.state_dict(), 'BestEvauationModel.pth')

            # BackWard
            losses.backward()

            # Update
            self.optimizer.step()
    
            if DEBUG :
                print(f'Batch {batch_idx + 1}/{len(self.data_loader_val)} in eval Loss: {loss_value:.4f}')
                break
        # self.lr_scheduler.step()
               
    def pridicte_and_display(self,predicte_path_csv=None):
        '''
        Function to prdict the output and display it with golden output 
        each image displaied in 5 subimage 6 labels displaied in each subimage 
        the golden output is dashed and the predicted is solid
        input:
            predicte_path_csv: string => path to data csv to predict
        output:
            diplay images 
        '''
        if predicte_path_csv==None:
            prdicted_dataloader=self.data_loader_val
        else:
            # create dataset
            prdicted_data = CustomDataset(dataset_path= predicte_path_csv, transform_type='val')
            # create data loader
            prdicted_dataloader = DataLoader(dataset=prdicted_data, batch_size=1, shuffle=False, num_workers=4)
        # make model in evaluation mode
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(prdicted_dataloader):
            # images = images.unsqueeze(1)
            images = torch.stack([image.to(self.device) for image in images])
            # convert targets from dic with keys: boxes, labels Only to list[Dict[str, Tensor]]
            targetdata=[]
            for i in range(len(images)):
                newdic={}
                newdic['boxes']=targets['boxes'][i].to(self.device)
                newdic['labels']=targets['labels'][i].to(self.device)
                targetdata.append(newdic)
            # forward
            with torch.no_grad():
                prediction = self.model(images)[1]
                print("scores",prediction[0]["scores"])
                # move image to cpu
                images = list(image.to(torch.device('cpu')) for image in images)
                for pred,targ,img in zip(prediction,targetdata,images):
                    # move the prediction to cpu
                    pred = {k: v.to(torch.device('cpu')) for k, v in pred.items()}
                    # move target to cpu
                    targ = {k: v.to(torch.device('cpu')) for k, v in targ.items()} 
                    boxes,labels=pred["boxes"], pred["labels"]
                    plot_image(img, targ["labels"].tolist(),targ["boxes"].tolist(),labels,boxes)
            break
                
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
                    edgecolor=region_colors[((i-j*6-1)%7)-1],
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

if __name__ == '__main__':
    object_detector_model=ObjectDetector().create_model()
    
    # trainer = Object_detector_trainer(model= torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=30))
    trainer = Object_detector_trainer(model=object_detector_model)
    # trainer = Object_detector_trainer()
    trainer.train(rpn_only=True)
    trainer.train()
    # trainer.evaluate()
    trainer.pridicte_and_display()