import torch
import torch.optim as optim
from torch.utils.data import  DataLoader

from src.x_reporto.data_loader.custom_dataset import CustomDataset

from src.x_reporto.models.x_reporto_factory import XReporto

from config import ModelStage,MODEL_STAGE,DEVICE
import gc

import sys
# constants
EPOCHS=1
LEARNING_RATE=0.0001
BATCH_SIZE=1
SCHEDULAR_STEP_SIZE=1
SCHEDULAR_GAMMA=0.9999999999
DEBUG=True


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
                self.load_model('object_detector',EPOCHS)
            elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                self.load_model('object_detector_classifier',EPOCHS)
            elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                self.load_model('LM',EPOCHS)
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
        self.eval_best_loss = float('inf')

    def train(self):
        '''
        Function to train X-Reporto training dataset depending on the MODEL_STAGE
        '''
        # make model in training mode
        self.model.train()

        for epoch in range(EPOCHS):
            for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(self.data_loader_train):
                # Check GPU memory usage
                print(torch.cuda.memory_allocated())
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
                abnormal_classifier_targets=None
                if MODEL_STAGE==ModelStage.CLASSIFIER.value :
                    # Selection
                    # Moving Object Detector Targets to Device
                    abnormal_classifier_targets=[]
                    for i in range(len(images)):
                        bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal'][i]
                        abnormal_classifier_targets.append(bbox_is_abnormal)
                    abnormal_classifier_targets=torch.stack(abnormal_classifier_targets).to(DEVICE)
                #     del abnormal_classifier_batch
                # # Check GPU memory usage
                # print(torch.cuda.memory_allocated())
                # torch.cuda.empty_cache()
                # print(gc.collect() )  
                # # Check GPU memory usage
                # print(torch.cuda.memory_allocated())
                # sys.exit()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward Pass
                # print(torch.cuda.memory_allocated())
                object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses= self.model(images, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets)   
                # del images
                # del object_detector_targets
                # del selection_classifier_targets
                # del abnormal_classifier_targets
                # Check GPU memory usage
                # print(torch.cuda.memory_allocated())
                # torch.cuda.empty_cache()
                # print(gc.collect() )  
                # # Check GPU memory usage
                # print(torch.cuda.memory_allocated())

                #backward pass
                Total_loss=None
                object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                Total_loss=object_detector_losses_summation
                if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                    Total_loss+=selection_classifier_losses
                    Total_loss+=abnormal_binary_classifier_losses
                Total_loss.backward()

                # update the parameters
                self.optimizer.step()

                if DEBUG :
                     # save the best model
                    if(Total_loss<self.best_loss):
                        self.best_loss=Total_loss
                        if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                            self.save_model('object_detector',epoch)
                        elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                            self.save_model('object_detector_classifier',epoch)
                        elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                            self.save_model('LM',epoch)
                        
                    print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation.item():.4f} selection_classifier_Loss: {selection_classifier_losses.item():.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses.item():.4f} total_Loss: {Total_loss.item():.4f}')
                    break
            # # update the learning rate
            # self.lr_scheduler.step()
                
                
                # print(images)
                # print(images.shape)
                # sys.exit()
    def save_model(self,name,epoch):
        torch.save(self.model.state_dict(), "models/"+name+str(epoch)+".pth")
    def load_model(self,name,epoch):
        self.model.load_state_dict(torch.load("models/"+name+str(epoch)+".pth"))




      
    

if __name__ == '__main__':
    x_reporto_model=XReporto().create_model()
    # print(x_reporto_model)
    
    trainer = XReportoTrainer(model= x_reporto_model)
    trainer.train()
    # trainer.evaluate()
    # trainer.pridicte_and_display()