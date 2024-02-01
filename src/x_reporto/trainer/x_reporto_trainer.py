import torch
import torch.optim as optim
from torch.utils.data import  DataLoader

from src.x_reporto.data_loader.custom_dataset import CustomDataset

from src.x_reporto.models.x_reporto_factory import XReporto
from src.utils import plot_image
from config import *

import gc
import sys
# constants

class XReportoTrainer():
    def __init__(self,training_csv_path: str='datasets/train.csv',validation_csv_path:str ='datasets/train.csv',
                 model=None):
        '''
        inputs:
            training_csv_path [string]: the path to the training csv file
            validation_csv_path [string]: the path to the validation csv file
            model: the x_reporto model
        '''

        # Model
        if model==None:
            # load the model from 
            self.model=XReporto().create_model()

            # TODO Fix Paths
            if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                self.load_model('object_detector')
            elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                self.load_model('object_detector_classifier')
            elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                self.load_model('LM')
        else:
            self.model = model
         
        self.model.to(DEVICE)

         # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= LEARNING_RATE)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)

        # create dataset
        # TODO Change to transform_type train
        self.dataset_train = CustomDataset(dataset_path= training_csv_path, transform_type='val')
        self.dataset_val = CustomDataset(dataset_path= validation_csv_path, transform_type='val')
        
        # create data loader
        # TODO @Ahmed Hosny suffle Training Loaders
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        self.data_loader_val = DataLoader(dataset=self.dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # initialize the best loss to a large value
        self.best_loss = float('inf')
        # self.best_loss = 0.3904
        self.eval_best_loss = float('inf')

    def train(self):
        '''
        Function to train X-Reporto training dataset depending on the MODEL_STAGE
        '''
        # make model in training mode
        self.model.train()
        epoch_loss = 0
        for epoch in range(EPOCHS):
            for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(self.data_loader_train):
                # Check GPU memory usage
                # print(torch.cuda.memory_allocated())
                images=object_detector_batch['image']

                # Move images to Device
                images = torch.stack([image.to(DEVICE) for image in images])
                # Check GPU memory usage
                # print(torch.cuda.memory_allocated())
                # Moving Object Detector Targets to Device
                object_detector_targets=[]
                for i in range(len(images)):
                    new_dict={}
                    new_dict['boxes']=object_detector_batch['bboxes'][i].to(DEVICE)
                    new_dict['labels']=object_detector_batch['bbox_labels'][i].to(DEVICE)
                    object_detector_targets.append(new_dict)
                    
                selection_classifier_targets=None
                abnormal_classifier_targets=None
                if MODEL_STAGE==ModelStage.CLASSIFIER.value :
                    # Selection
                    # Moving Object Detector Targets to Device
                    selection_classifier_targets=[]
                    for i in range(len(images)):
                        phrase_exist=selection_classifier_batch['bbox_phrase_exists'][i]
                        selection_classifier_targets.append(phrase_exist)
                    selection_classifier_targets=torch.stack(selection_classifier_targets).to(DEVICE)
                    # del selection_classifier_batch

                    # Abnormal
                    # Selection
                    # Moving Object Detector Targets to Device
                    abnormal_classifier_targets=[]
                    for i in range(len(images)):
                        bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal'][i]
                        abnormal_classifier_targets.append(bbox_is_abnormal)
                    abnormal_classifier_targets=torch.stack(abnormal_classifier_targets).to(DEVICE)
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward Pass
                # print(torch.cuda.memory_allocated())
                object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses= self.model(images, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets)   

                # Free GPU memory 
                del object_detector_targets
                del selection_classifier_targets
                del abnormal_classifier_targets
                del images
                torch.cuda.empty_cache()

                #backward pass
                Total_loss=None
                object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                Total_loss=object_detector_losses_summation.clone()
                if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                    Total_loss+=selection_classifier_losses
                    Total_loss+=abnormal_binary_classifier_losses

                Total_loss.backward()

                epoch_loss += Total_loss
                # update the parameters
                self.optimizer.step()

                if DEBUG :
                    print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} total_Loss: {Total_loss:.4f}')
                    
                    # free gpu memory
                    del Total_loss
                    del object_detector_losses
                    del selection_classifier_losses
                    del abnormal_binary_classifier_losses
                    torch.cuda.empty_cache()
                    
                    # break
            # save the best model
            if(epoch_loss<self.best_loss):
                self.best_loss=epoch_loss
                if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                    self.save_model('object_detector')
                elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                    self.save_model('object_detector_classifier')
                elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    self.save_model('LM')
                # wirte the loss to a file
                with open("loss.txt", "a") as myfile:
                    myfile.write(f'epoch: {epoch+1}/{EPOCHS}, epoch loss: {epoch_loss/len(self.data_loader_train):.4f}')
                    myfile.write("\n")
                # print the epoch loss
                print("\n")
                print(f'epoch: {epoch+1}/{EPOCHS}, epoch loss: {epoch_loss/len(self.data_loader_train):.4f}')
                print("\n")
            epoch_loss=0

            # # update the learning rate
            # self.lr_scheduler.step()
    def Valdiate(self):
        '''
        Function to evaluate the object detector on evaluation dataset
        '''
        # make model in training mode
        self.model.eval()
        with torch.no_grad():
        
            for epoch in range(EPOCHS):
                for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(self.data_loader_val):
                    # Check GPU memory usage
                    # print(torch.cuda.memory_allocated())
                    
                    images=object_detector_batch['image']

                    # Move images to Device
                    images = torch.stack([image.to(DEVICE) for image in images])
                    # Check GPU memory usage
                    # print(torch.cuda.memory_allocated())
                    # Moving Object Detector Targets to Device
                    object_detector_targets=[]
                    for i in range(len(images)):
                        new_dict={}
                        new_dict['boxes']=object_detector_batch['bboxes'][i].to(DEVICE)
                        new_dict['labels']=object_detector_batch['bbox_labels'][i].to(DEVICE)
                        object_detector_targets.append(new_dict)
                    # del object_detector_batch
                        
                    selection_classifier_targets=None
                    abnormal_classifier_targets=None
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value :
                        # Selection
                        # Moving Object Detector Targets to Device
                        selection_classifier_targets=[]
                        for i in range(len(images)):
                            phrase_exist=selection_classifier_batch['bbox_phrase_exists'][i]
                            selection_classifier_targets.append(phrase_exist)
                        selection_classifier_targets=torch.stack(selection_classifier_targets).to(DEVICE)
                        # del selection_classifier_batch

                        # Abnormal
                        # Selection
                        # Moving Object Detector Targets to Device
                        abnormal_classifier_targets=[]
                        for i in range(len(images)):
                            bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal'][i]
                            abnormal_classifier_targets.append(bbox_is_abnormal)
                        abnormal_classifier_targets=torch.stack(abnormal_classifier_targets).to(DEVICE)
                    # Forward Pass
                    object_detector_losses,_,_,selection_classifier_losses,_,abnormal_binary_classifier_losses,_= self.model(images, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets)   

                    # Free GPU memory 
                    del object_detector_targets
                    del selection_classifier_targets
                    del abnormal_classifier_targets
                    del images
                    torch.cuda.empty_cache()

                    Total_loss=None
                    object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                    Total_loss=object_detector_losses_summation.clone()
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                        Total_loss+=selection_classifier_losses
                        Total_loss+=abnormal_binary_classifier_losses

                    if DEBUG :
                        print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} total_Loss: {Total_loss:.4f}')
                        
                        # free gpu memory
                        del Total_loss
                        del object_detector_losses
                        del selection_classifier_losses
                        del abnormal_binary_classifier_losses
                        torch.cuda.empty_cache()
                        
                        # break
   
    def predict_and_display(self,predict_path_csv=None):
        '''
        Function to prdict the output and display it with golden output 
        each image displaied in 5 subimage 6 labels displaied in each subimage 
        the golden output is dashed and the predicted is solid
        input:
            predicte_path_csv: string => path to data csv to predict
        output:
            diplay images 
        '''
        if predict_path_csv==None:
                predicted_dataloader=self.data_loader_val
        else:
                predicted_data = CustomDataset(dataset_path= predict_path_csv, transform_type='val')
                # create data loader
                predicted_dataloader = DataLoader(dataset=predicted_data, batch_size=1, shuffle=False, num_workers=4)
                # make model in training mode
        self.model.eval()
        with torch.no_grad():
        
            for epoch in range(EPOCHS):
                for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(predicted_dataloader):
                    # Check GPU memory usage
                    # print(torch.cuda.memory_allocated())
                    
                    images=object_detector_batch['image']

                    # Move images to Device
                    images = torch.stack([image.to(DEVICE) for image in images])
                    # Check GPU memory usage
                    # print(torch.cuda.memory_allocated())
                    # Moving Object Detector Targets to Device
                    object_detector_targets=[]
                    for i in range(len(images)):
                        new_dict={}
                        new_dict['boxes']=object_detector_batch['bboxes'][i].to(DEVICE)
                        new_dict['labels']=object_detector_batch['bbox_labels'][i].to(DEVICE)
                        object_detector_targets.append(new_dict)
                    # del object_detector_batch
                        
                    selection_classifier_targets=None
                    abnormal_classifier_targets=None
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value :
                        # Selection
                        # Moving Object Detector Targets to Device
                        selection_classifier_targets=[]
                        for i in range(len(images)):
                            phrase_exist=selection_classifier_batch['bbox_phrase_exists'][i]
                            selection_classifier_targets.append(phrase_exist)
                        selection_classifier_targets=torch.stack(selection_classifier_targets).to(DEVICE)
                        # del selection_classifier_batch

                        # Abnormal
                        # Selection
                        # Moving Object Detector Targets to Device
                        abnormal_classifier_targets=[]
                        for i in range(len(images)):
                            bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal'][i]
                            abnormal_classifier_targets.append(bbox_is_abnormal)
                        abnormal_classifier_targets=torch.stack(abnormal_classifier_targets).to(DEVICE)
                    # Forward Pass
                    object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,_,abnormal_binary_classifier_losses,_= self.model(images, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets) 
                    # object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions
  
                    plot_image(images[0].cpu(),object_detector_targets[0]["labels"].tolist(),object_detector_targets[0]["boxes"].tolist(),object_detector_detected_classes[0].tolist(),object_detector_boxes[0].tolist())
                    

                    # Free GPU memory 
                    del object_detector_targets
                    del selection_classifier_targets
                    del abnormal_classifier_targets
                    del images
                    torch.cuda.empty_cache()

                    Total_loss=None
                    object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                    Total_loss=object_detector_losses_summation.clone()
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                        Total_loss+=selection_classifier_losses
                        Total_loss+=abnormal_binary_classifier_losses

                    if DEBUG :
                        print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} total_Loss: {Total_loss:.4f}')
                        
                        # free gpu memory
                        del Total_loss
                        del object_detector_losses
                        del selection_classifier_losses
                        del abnormal_binary_classifier_losses
                        torch.cuda.empty_cache()
                        
                        # break

    # make model in evaluation mode
    def save_model(self,name):
        torch.save(self.model.state_dict(), "models/"+name+".pth")
    def load_model(self,name):
        self.model.load_state_dict(torch.load("models/"+name+".pth"))




      
    

if __name__ == '__main__':
    # x_reporto_model=XReporto().create_model()
    
    # trainer = XReportoTrainer(model= x_reporto_model)
    trainer = XReportoTrainer()
    trainer.train()
    # trainer.Valdiate()
    # trainer.predict_and_display(predict_path_csv='datasets/predict.csv')