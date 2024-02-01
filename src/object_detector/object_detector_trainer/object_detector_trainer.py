import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from ..data_loader.custom_dataset import CustomDataset
from matplotlib import patches
from src.object_detector.models.object_detector_factory import ObjectDetector
import os

# constants
EPOCHS=30
LEARNING_RATE=0.0000005
BATCH_SIZE=1
SCHEDULAR_STEP_SIZE=500
SCHEDULAR_GAMMA=0.9999999999999999
DEBUG=True


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
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
            if os.path.exists('bestmodel.pth'):
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
        self.data_loader_train = DataLoader(dataset=self.dataset_train, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        self.data_loader_val = DataLoader(dataset=self.dataset_val, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # initialize the best loss to a large value
        self.bestloss=100000
        self.evalbestloss=100000
    
    def train(self,rpn_only=False):
        '''
        Function to train the object detector on training dataset
        '''
        # make model in trainning mode
        self.model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_idx, (images, targets) in enumerate(self.data_loader_train):
                # add new dimension to images after batch size
                images = torch.stack([image.to(self.device) for image in images])
                # convert targets from dic with keys: boxes, labels Only to list[Dict[str, Tensor]]
                targetdata=[]
                for i in range(len(images)):
                    newdic={}
                    newdic['boxes']=targets[i]['boxes'].to(self.device)
                    newdic['labels']=targets[i]['labels'].to(self.device)
                    targetdata.append(newdic)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss_dict,_= self.model(images, targetdata)   
                del targetdata
                del images
                torch.cuda.empty_cache()
                
                losses = sum(loss for loss in loss_dict.values())

                # sum rpn losses
                if rpn_only:
                    losses = loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']
                loss_value = losses.item()
                total_loss += loss_value

                # BackWard
                losses.backward()

                # Update
                self.optimizer.step()
        
                if DEBUG :
                    print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} Loss: {loss_value:.4f}')
                    
            # self.lr_scheduler.step()
            # save the best model
            if(total_loss<self.bestloss):
                self.bestloss=total_loss
                torch.save(self.model.state_dict(), 'bestmodel.pth')
                print(f'epoch: {epoch+1}, total_loss: {total_loss/len(self.data_loader_train):.4f}')
            # free Gpu memory
            torch.cuda.empty_cache()

    def evaluate(self):
        '''
        Function to evaluate the object detector on evaluation dataset
        '''

        # make the model in evaluation mode
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(self.data_loader_val):
            # add new dimension to images after batch size
            # images = images.unsqueeze(1)
            images = torch.stack([image.to(self.device) for image in images])
            # convert targets from dic with keys: boxes, labels Only to list[Dict[str, Tensor]]
            targetdata=[]
            for i in range(len(images)):
                newdic={}
                newdic['boxes']=targets[i]['boxes'].to(self.device)
                newdic['labels']=targets[i]['labels'].to(self.device)
                targetdata.append(newdic)
            with torch.no_grad():
                # forward + backward + optimize
                loss_dict,prediction= self.model(images, targetdata)   
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                
            # save the best model
            if(loss_value<self.evalbestloss):
                self.evalbestloss=loss_value
                torch.save(self.model.state_dict(), 'BestEvauationModel.pth')
    
            if DEBUG :
                print(f'Batch {batch_idx + 1}/{len(self.data_loader_val)} in eval Loss: {loss_value:.4f}')
                               
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
            prdicted_dataloader = DataLoader(dataset=prdicted_data, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=4)
        # make model in evaluation mode
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(prdicted_dataloader):
            # images = images.unsqueeze(1)
            images = torch.stack([image.to(self.device) for image in images])
            # convert targets from dic with keys: boxes, labels Only to list[Dict[str, Tensor]]
            targetdata=[]
            for i in range(len(images)):
                newdic={}
                newdic['boxes']=targets[i]['boxes'].to(self.device)
                newdic['labels']=targets[i]['labels'].to(self.device)
                targetdata.append(newdic)
            # forward
            with torch.no_grad():
                loses,prediction = self.model(images)
                boxes=prediction["detections"]["top_region_boxes"][0].tolist()
                labels=prediction["class_detected"][0].tolist()
                # move image to cpu
                images = list(image.to(torch.device('cpu')) for image in images)
                for targ,img in zip(targetdata,images):
                    # move the prediction to cpu
                    # move target to cpu
                    targ = {k: v.to(torch.device('cpu')) for k, v in targ.items()} 
                    plot_image(img, targ["labels"].tolist(),targ["boxes"].tolist(),labels,boxes)
            
                
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

def compute_IOU(pred_box, target_box):
    '''
    Function to compute the Intersection over Union (IOU) of two boxes.

    inputs:

        pred_box: predicted box (Format [xmin, ymin, xmax, ymax])
        target_box: target box (Format [xmin, ymin, xmax, ymax])
    '''
    if pred_box is None or target_box is None:
        return 0

    # compute the intersection area
    x1 = max(pred_box[0], target_box[0])
    y1 = max(pred_box[1], target_box[1])
    x2 = min(pred_box[2], target_box[2])
    y2 = min(pred_box[3], target_box[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # compute the union area
    pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    target_box_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    union_area = pred_box_area + target_box_area - intersection_area

    # compute the IOU 0 (no overlap) -> 1 totally overlap
    iou = intersection_area / union_area
    return iou

def compute_precision(pred_boxes,pred_labels, target_boxes,target_labels, iou_threshold=0.5):
    '''
    Function to compute the precision.

    inputs:
        pred_boxes: list of predicted boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
        pred_labels: list of predicted labels (Format [N] => N times label)
        target_boxes: list of target boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
        target_labels: list of target labels (Format [N] => N times label)
        iou_threshold: threshold to consider a prediction to be correct
    '''
    # compute the number of predicted boxes
    num_pred_boxes = len(pred_boxes)
    # compute the number of target boxes
    num_target_boxes = len(target_boxes)
    # compute the number of true positive detections
    num_true_positive = 0
    num_false_positive = 0
    num_false_negative = 0
    index = 1
    # for each predicted box
    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        # for each target box
        if pred_label != 0 and index in target_labels:
            if compute_IOU(pred_box, target_boxes[index-1]) > iou_threshold:
                # increment the number of true positive detections
                num_true_positive += 1
            else:
                num_false_positive += 1
        elif pred_label != 0 and index not in target_labels:
            num_false_positive += 1
        elif pred_label == 0 and index in target_labels:
            num_false_negative += 1
        # increment the index
        index += 1            
    # compute the precision
    precision = num_true_positive / (num_true_positive+num_false_positive)

    # compute the recall
    recall = num_true_positive / (num_true_positive+num_false_negative)

    F1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f'precision: {precision:.4f}, recall: {recall:.4f}, F1_score: {F1_score:.4f}')

    # return the precision and recall
    return precision, recall, F1_score


if __name__ == '__main__':
    object_detector_model=ObjectDetector().create_model()    
    trainer = Object_detector_trainer(model=object_detector_model)

    # trainer = Object_detector_trainer()
    
    # trainer.train(rpn_only=True)
    trainer.train()
    # trainer.evaluate()
    # trainer.pridicte_and_display(predicte_path_csv='datasets/predict.csv')