# Logging
from logger_setup import setup_logging
import logging


import os
import gc
from tqdm import tqdm


# Torch
import torch
import torch.optim as optim
from torch.utils.data import  DataLoader

# Modules
from src.x_reporto.models.x_reporto_factory import XReporto
from src.x_reporto.data_loader.custom_dataset import CustomDataset

# Utils 
from src.utils import plot_image,save_model,save_checkpoint,load_checkpoint,seed_worker

from config import *

class XReportoTrainer():
    """
    XReportoTrainer class is responsible for training, validating, and predicting with the X-Reporto model.

    Args:
        training_csv_path (str): Path to the training CSV file.
        validation_csv_path (str): Path to the validation CSV file.
        model Optional[XReporto]: The X-Reporto model.If not provided, the model is loaded from a .pth file

    Methods:
        - train(): Train the X-Reporto model depending on the MODEL_STAGE.
        - validate(): Evaluate the object detector on the validation dataset.
        - predict_and_display(predict_path_csv=None): Predict the output and display it with golden output.
        - save_model(name): Save the current state of the X-Reporto model.
        - load_model(name): Load a pre-trained X-Reporto model.
    
    Examples:
        >>> x_reporto_model = XReporto().create_model()

        >>> # Create an XReportoTrainer instance with the X-Reporto model
        >>> trainer = XReportoTrainer(model=x_reporto_model)

        >>> # Alternatively, create an XReportoTrainer instance without specifying the model
        >>> trainer = XReportoTrainer()

        >>> # Train the X-Reporto model on the training dataset
        >>> trainer.train()

        >>> # Run Validation
        >>> trainer.validate()

        >>> # Predict and display results
        >>> trainer.predict_and_display(predict_path_csv='datasets/predict.csv')
    """
    def __init__(self, model:XReporto,training_csv_path: str =training_csv_path,validation_csv_path:str = validation_csv_path):
        '''
        inputs:
            training_csv_path (str): the path to the training csv file
            validation_csv_path (str): the path to the validation csv file
            model Optional[XReporto]: the x_reporto model instance to be trained.If not provided, the model is loaded from a .pth file.
        '''
        self.model = model

        # Move to device
        self.model.to(DEVICE)

        # create adam optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr= LEARNING_RATE, weight_decay=0.0005)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=SCHEDULAR_GAMMA, patience=SCHEDULAR_STEP_SIZE, threshold=THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)

        # create dataset
        self.dataset_train = CustomDataset(dataset_path= training_csv_path, transform_type='train')
        logging.info("Train dataset loaded")
        self.dataset_val = CustomDataset(dataset_path= validation_csv_path, transform_type='val')
        logging.info("Validation dataset loaded")

        # create data loader
        g = torch.Generator()
        g.manual_seed(SEED)
        self.data_loader_train = DataLoader(dataset=self.dataset_train,collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
        logging.info(f"Training DataLoader Loaded Size: {len(self.data_loader_train)}")
      
        self.data_loader_val = DataLoader(dataset=self.dataset_val, collate_fn=collate_fn,batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        logging.info(f"Validation DataLoader Loaded Size: {len(self.data_loader_val)}")

        # initialize the best loss to a large value
        self.best_loss = float('inf')


    def test_data_loader(self):
        '''
        Test the data loader by iterating over the training dataset and printing the length of each batch.
        '''
        for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets) in tqdm(enumerate(self.data_loader_train)):
            # check the length of each batch using assert with print statements
            # check that object_detector_targets dictionary boxes and labels have the same length
            assert len(object_detector_targets[0]['boxes']) == len(object_detector_targets[0]['labels']) , f'Batch {batch_idx + 1} has different number of boxes and labels'
            # assert that boxes is shape (N,4)
           
            for i in range(len(images)):
                assert object_detector_targets[i]['boxes'].shape[1] == 4, f'Batch {batch_idx + 1} has boxes with shape {object_detector_targets[i]["boxes"].shape}'
            # print(f'Batch {batch_idx + 1} has {len(images)} images')
            logging.info(f'Batch {batch_idx + 1} has {len(images)} images')

    def train(self,start_epoch=0,epoch_loss_init=0,start_batch=0):
        '''
        Train X-Reporto on the training dataset depending on the MODEL_STAGE.
        '''
        # make model in training mode
        logging.info("Start Training")
        self.model.train()
        total_steps=0
        # for epoch in range(EPOCHS):
        for epoch in range(start_epoch, start_epoch + EPOCHS):
            if epoch==start_epoch:
                # Load loss from chkpt
                epoch_loss=epoch_loss_init
            else:
                epoch_loss = 0
            for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets) in enumerate(self.data_loader_train):

                if batch_idx < start_batch:
                    continue  # Skip batches until reaching the desired starting batch number

                # Test Recovery
                # if epoch==3 and batch_idx==1:
                    # print("Start Next time from")
                    # print(object_detector_targets[1])
                    # print(batch_idx)
                    # raise Exception("CRASSSSSSSSSSSSHHHHHHHHHHHHHHHHHHHHHHH")         
                # TODO: Test
                # Move inputs to Device
                # print(object_detector_targets[0])
                # print(batch_idx)

                images = images.to(DEVICE)
                object_detector_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in object_detector_targets]

                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Selection Classifier
                    # Moving Selection Classifier Targets to Device
                    selection_classifier_targets = selection_classifier_targets.to(DEVICE)
                    abnormal_classifier_targets = abnormal_classifier_targets.to(DEVICE)
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    # Language Model
                    # Moving Language Model Targets to Device
                    LM_targets = LM_targets.to(DEVICE)
                    input_ids = LM_inputs['input_ids'].to(DEVICE)
                    attention_mask = LM_inputs['attention_mask'].to(DEVICE)
                    loopLength= input_ids.shape[1]
                    Total_loss=self.language_model_forward_pass(images=images,input_ids=input_ids,attention_mask=attention_mask,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets,LM_targets=LM_targets,loopLength=loopLength,LM_Batch_Size=LM_Batch_Size)
                    epoch_loss += Total_loss
                else:
                    Total_loss=self.object_detector_and_classifier_forward_pass(images=images,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets)
                    epoch_loss += Total_loss

                # backward pass
                Total_loss.backward()
                # Acculmulation Learning
                if (batch_idx+1) %ACCUMULATION_STEPS==0:
                    # update the parameters
                    self.optimizer.step()
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    logging.debug(f' Update Weights at  epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} ')
                    
                # update the learning rate
                self.lr_scheduler.step(Total_loss)
                # Get the new learning rate
                new_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_train)}, Learning Rate: {new_lr:.10f}")

                if (batch_idx+1)%100==0:
                    # Every 100 Batch print Average Loss for epoch till Now
                    logging.info(f'[Every 100 Batch]: Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(self.data_loader_train)}, Average Cumulative Epoch Loss : {epoch_loss/(batch_idx+1):.4f}')

                # Checkpoint
                total_steps+=1            
                if(total_steps%CHECKPOINT_EVERY_N ==0):
                    save_checkpoint(epoch=epoch,batch_index=batch_idx,optimizer_state=self.optimizer.state_dict(),
                                    scheduler_state_dict=self.lr_scheduler.state_dict(),model_state=self.model.state_dict(),
                                    best_loss=self.best_loss,epoch_loss=epoch_loss)
                    total_steps=0
               
            logging.info(f'Epoch {epoch+1}/{EPOCHS}, Total epoch Loss: {epoch_loss:.4f} Average epoch loss : {epoch_loss/(len(self.data_loader_train)):.4f}')
            # Free GPU memory
            del Total_loss
            torch.cuda.empty_cache()
            gc.collect()
            # validate the model
            self.model.eval()
            validation_loss= self.validate_during_training()  #sechdule the learning rate
            self.model.train()

            # saving model per epoch
            if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                if TRAIN_RPN:
                    # Saving object_detector marked as rpn
                    self.save_model(model=self.model.object_detector,name="object_detector_rpn",epoch=epoch,validation_loss=validation_loss)
                else:
                     self.save_model(model=self.model.object_detector,name="object_detector",epoch=epoch,validation_loss=validation_loss)
            elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                # Save Region Selection Classifier
                self.save_model(model=self.model.binary_classifier_selection_region,name="region_classifier",epoch=epoch,validation_loss=validation_loss)
                # Save Abnormal Classifier
                self.save_model(model=self.model.binary_classifier_region_abnormal,name="abnormal_classifier",epoch=epoch,validation_loss=validation_loss)
            elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                #Save language model
                self.save_model(model=self.model.language_model,name='LM',epoch=epoch,validation_loss=validation_loss)                               
        # save the best model            
        logging.info("Training Done")
    
    def save_model(self,model:torch.nn.Module,name:str,epoch:int,validation_loss:float):
        '''
        Save the current state of model.
        '''
        logging.info("Saving "+name+"_epoch "+str(epoch+1))
        save_model(model=model,name=name+"_epoch_"+str(epoch+1))
        self.check_best_model(epoch,validation_loss,name,model)  

    def check_best_model(self,epoch:int,validation_loss:float,name:str,model:torch.nn.Module):
        '''
        Check if the current model is the best model
        '''
        if(validation_loss<self.best_loss) :
                self.best_loss=validation_loss
                save_model(model=model,name=name+"_best")
                logging.info(f"Best Model Updated: {name}_best at epoch {epoch+1} with Average validation loss: {self.best_loss:.4f}")

    def  object_detector_and_classifier_forward_pass(self,images:torch.Tensor,object_detector_targets:torch.Tensor,selection_classifier_targets:torch.Tensor,abnormal_classifier_targets:torch.Tensor,validate_during_training:bool=False):
        # Forward Pass
        object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses= self.model(images=images,input_ids=None,attention_mask=None,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets,validate_during_training=validate_during_training)
        
        Total_loss=None
        object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
        Total_loss=object_detector_losses_summation.clone()
        if MODEL_STAGE==ModelStage.CLASSIFIER.value:
            Total_loss+=selection_classifier_losses
            Total_loss+=abnormal_binary_classifier_losses
        del LM_losses
        del object_detector_losses
        del selection_classifier_losses
        del abnormal_binary_classifier_losses
        torch.cuda.empty_cache()
        gc.collect()
        return Total_loss

    def language_model_forward_pass(self,images:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor,object_detector_targets:torch.Tensor,selection_classifier_targets:torch.Tensor,abnormal_classifier_targets:torch.Tensor,LM_targets:torch.Tensor,epoch:int,batch_idx:int,loopLength:int,LM_Batch_Size:int,validate_during_training:bool=False):
        for batch in range(BATCH_SIZE):
            total_LM_losses=0
            for i in range(0,loopLength,LM_Batch_Size):

                # Forward Pass
                object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,stop= self.model(images=images,input_ids=input_ids,attention_mask=attention_mask,object_detector_targets= object_detector_targets,selection_classifier_targets= selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets,LM_targets=LM_targets,batch=batch,index=i,validate_during_training=validate_during_training)

                if stop:
                    break
                # Backward pass
                Total_loss=None
                object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                Total_loss=object_detector_losses_summation.clone()
                Total_loss+=selection_classifier_losses
                Total_loss+=abnormal_binary_classifier_losses
                Total_loss+=LM_losses
                total_LM_losses+=LM_losses

                logging.debug(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} LM_losses: {total_LM_losses:.4f} total_Loss: {object_detector_losses_summation+selection_classifier_losses+abnormal_binary_classifier_losses+total_LM_losses:.4f}')

            # Free GPU memory
            del LM_losses
            del object_detector_losses
            del selection_classifier_losses
            del abnormal_binary_classifier_losses
            torch.cuda.empty_cache()
            gc.collect()
        return Total_loss

    def validate_during_training(self):
        '''
        Validate the model during training
        '''
        with torch.no_grad():
            validation_total_loss=0
            total_loss=0
            for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets) in enumerate(self.data_loader_val):
                # Move inputs to Device
                images = images.to(DEVICE)
                object_detector_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in object_detector_targets]
                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Selection Classifier
                    # Moving Selection Classifier Targets to Device
                    selection_classifier_targets = selection_classifier_targets.to(DEVICE)
                    abnormal_classifier_targets = abnormal_classifier_targets.to(DEVICE)
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    # Language Model
                    # Moving Language Model Targets to Device
                    LM_targets = LM_targets.to(DEVICE)
                    input_ids = LM_inputs['input_ids'].to(DEVICE)
                    attention_mask = LM_inputs['attention_mask'].to(DEVICE)
                    loopLength= input_ids.shape[1]
                    validation_total_loss+=self.language_model_forward_pass(images=images,input_ids=input_ids,attention_mask=attention_mask,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets,LM_targets=LM_targets,loopLength=loopLength,LM_Batch_Size=LM_Batch_Size,validate_during_training=True)
                else:
                    total_loss=self.object_detector_and_classifier_forward_pass(images=images,object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,abnormal_classifier_targets=abnormal_classifier_targets,validate_during_training=True)
                    validation_total_loss+=total_loss
            # arverge validation_total_loss
            validation_total_loss/=(len(self.data_loader_train))
            # update the learning rate according to the validation loss if decrease
            self.lr_scheduler.step(validation_total_loss)
            return validation_total_loss
            
            
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    image_shape = batch[0][0]["image"].size()
    images = torch.empty(size=(len(batch), *image_shape))
    object_detector_targets=[]
    selection_classifier_targets=[]
    abnormal_classifier_targets=[]
    LM_targets=[]
    input_ids=[]
    attention_mask=[]
    LM_inputs={}

    for i in range(len(batch)):
        (object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) = batch[i]
        # stack images
        images[i] = object_detector_batch['image']
        # Moving Object Detector Targets to Device
        new_dict={}
        new_dict['boxes']=object_detector_batch['bboxes']
        new_dict['labels']=object_detector_batch['bbox_labels']
        object_detector_targets.append(new_dict)
        
        bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal']
        abnormal_classifier_targets.append(bbox_is_abnormal)

        phrase_exist=selection_classifier_batch['bbox_phrase_exists']
        selection_classifier_targets.append(phrase_exist)

        phrase=LM_batch['label_ids']
        LM_targets.append(phrase)
        input_ids.append(LM_batch['input_ids'])
        attention_mask.append(LM_batch['attention_mask'])


    selection_classifier_targets=torch.stack(selection_classifier_targets)
    abnormal_classifier_targets=torch.stack(abnormal_classifier_targets)
    LM_targets=torch.stack(LM_targets)
    LM_inputs['input_ids']=torch.stack(input_ids)
    LM_inputs['attention_mask']=torch.stack(attention_mask)

    return images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets

def main():
    logging.info("Training X_Reporto Started")
    # Logging Configurations
    log_config()
    if OperationMode.TRAINING.value!=OPERATION_MODE :
        #throw exception 
        raise Exception("Operation Mode is not Training Mode")
    # Creating run folder
    folder_path="models/" + str(RUN)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"Folder '{folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{folder_path}' already exists.")

    # Creating checkpoints folder
    folder_path="check_points/" + str(RUN)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"Folder '{folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{folder_path}' already exists.")

    # X-Reporto Trainer Object
    x_reporto_model = XReporto().create_model()

    # Create an XReportoTrainer instance with the X-Reporto model
    trainer = XReportoTrainer(model=x_reporto_model)

    if RECOVER:
        # Load the state of model
        checkpoint=load_checkpoint(run=RUN)

        # Load Model state
        x_reporto_model.load_state_dict(checkpoint['model_state'])

        # Batch to start from
        start_batch=checkpoint['batch_index']+1

        # Load scheduler_state_dict
        trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load optimizer_state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Load best_loss
        trainer.best_loss=checkpoint['best_loss']


        # Start Train form checkpoint ends
        trainer.train(start_epoch=checkpoint['epoch'],epoch_loss_init=checkpoint['epoch_loss'].item(),start_batch=start_batch)

    else:
        # No check point
        # Start New Training
        trainer.train()
        

    
if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/x_reporto_trainer.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
