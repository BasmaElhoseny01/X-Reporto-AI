import torch
import torch.optim as optim
from torch.utils.data import  DataLoader

from src.x_reporto.data_loader.custom_dataset import CustomDataset

from src.x_reporto.models.x_reporto_factory import XReporto

from config import ModelStage,MODEL_STAGE,DEVICE

import sys
# constants
EPOCHS=1
LEARNING_RATE=0.0001
BATCH_SIZE=2
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
                state_dict=torch.load('bestmodel.pth')
            elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                state_dict=torch.load('bestmodel.pth')
            elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                state_dict=torch.load('bestmodel.pth')

            self.model.load_state_dict(state_dict)

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
            for batch_idx,batch in enumerate(self.data_loader_train):
                images=batch[0]['image']

                # Move images to Device
                images = torch.stack([image.to(DEVICE) for image in images])


                # Moving Object Detector Targets to Device
                object_detector_targets=[]
                for i in range(len(images)):
                    new_dict={}
                    new_dict['boxes']=batch[0]['bboxes'][i].to(DEVICE)
                    new_dict['labels']=batch[0]['bbox_labels'][i].to(DEVICE)
                    object_detector_targets.append(new_dict)

                # del batch[0]
                    
                selection_classifier_targets=None
                if MODEL_STAGE==ModelStage.CLASSIFIER.value :
                    # Selection
                    # Moving Object Detector Targets to Device
                    selection_classifier_targets=[]
                    for i in range(len(images)):
                        phrase_exist=batch[1]['bbox_phrase_exists'][i]
                        selection_classifier_targets.append(phrase_exist)
                    selection_classifier_targets=torch.stack(selection_classifier_targets).to(DEVICE)

                    # UpNormal

                # del batch[1]
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward Pass
                x= self.model(images, object_detector_targets ,selection_classifier_targets)   
                print(x)
                sys.exit()

                
                
                # print(images)
                # print(images.shape)
                # sys.exit()



      
    

if __name__ == '__main__':
    x_reporto_model=XReporto().create_model()
    # print(x_reporto_model)
    
    trainer = XReportoTrainer(model= x_reporto_model)
    trainer.train()
    # trainer.evaluate()
    # trainer.pridicte_and_display()