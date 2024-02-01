import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
# from data_loader import F_RCNNDataset
from matplotlib import patches

from src.object_detector.data_loader.custom_dataset import CustomDataset
# constants
EPOCHS=1
LEARNING_RATE=0.000001
BATCH_SIZE=1
SCHEDULAR_STEP_SIZE=1
SCHEDULAR_GAMMA=0.1
DEBUG=True

class Object_detector_trainer:
    
    def __init__(self,training_csv_path='datasets/train-200.csv',validation_csv_path='datasets/train-200.csv',
                 model=None):
        '''
        inputs:
            training_csv_path: string => the path to the training csv file
            validation_csv_path: string => the path to the validation csv file
            model: the object detector model
        '''
        # connect to gpu if available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.model.to(self.device)

        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= LEARNING_RATE)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)

        # create dataset
        self.dataset_train = CustomDataset(dataset_path= training_csv_path,transform_type='train')
        self.dataset_val = CustomDataset(dataset_path= validation_csv_path,transform_type='val')
        
        # create data loader
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        self.data_loader_val = DataLoader(dataset=self.dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # initialize the best loss to a large value
        self.bestloss=10000000
        self.evalbestloss=10000000
    
    def train(self):
        '''
        Function to train the object detector on training dataset
        '''
        # make model in trainning mode
        self.model.train()
        for epoch in range(EPOCHS):
            for batch_idx, (images, targets) in enumerate(self.data_loader_train):
                # add new dimension to images after batch size
                images = images.unsqueeze(1)
                images = list(image.to(self.device) for image in images)
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
                if(loss_value<self.bestloss):
                    self.bestloss=loss_value
                    torch.save(self.model.state_dict(), 'bestmodel.pth')

                # BackWard
                losses.backward()

                # Update
                self.optimizer.step()
        
                if DEBUG :
                    print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} Loss: {loss_value:.4f}')
            
            self.lr_scheduler.step()

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
  
    def evaluate(self):
        '''
        Function to evaluate the object detector on evaluation dataset
        '''

        # make the model in evaluation mode
        # self.model.eval()
        self.model.train()
        for batch_idx, (images, targets) in enumerate(self.data_loader_val):
            # add new dimension to images after batch size
            images = images.unsqueeze(1)
            images = list(image.to(self.device) for image in images)
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
            print("eval loss is :" , loss_dict)
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
        
        self.lr_scheduler.step()
               
    def pridicte_and_display(self,predicte_path_csv):
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
            prdicted_data = CustomDataset(dataset_path= predicte_path_csv,transform_type='val')
            
            # create data loader
            prdicted_dataloader = DataLoader(dataset=prdicted_data, batch_size=1, shuffle=False, num_workers=4)
        # make model in evaluation mode
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(prdicted_dataloader):
            images = images.unsqueeze(1)
            images = list(image.to(self.device) for image in images)
            # convert targets from dic with keys: boxes, labels Only to list[Dict[str, Tensor]]
            targetdata=[]
            for i in range(len(images)):
                newdic={}
                newdic['boxes']=targets['boxes'][i].to(self.device)
                newdic['labels']=targets['labels'][i].to(self.device)
                targetdata.append(newdic)
            # forward
            with torch.no_grad():
                prediction = self.model(images)[0]
                # move the prediction to cpu
                prediction = {k: v.to(torch.device('cpu')) for k, v in prediction.items()}
                # move image to cpu
                images = list(image.to(torch.device('cpu')) for image in images)
                # move target to cpu
                targetdata = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in targetdata]
                
                boxes,labels=self.get_top_k_boxes_for_labels(prediction["boxes"], prediction["labels"], prediction["scores"], k=1)
                plot_image(images[0], targetdata[0]["labels"].tolist(),targetdata[0]["boxes"].tolist(),labels,boxes)
                
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

if __name__ == '__main__':
    trainer = Object_detector_trainer(model= torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=30))
    trainer.train()
    print("finish training")
    trainer.evaluate()
    trainer.pridicte_and_display()