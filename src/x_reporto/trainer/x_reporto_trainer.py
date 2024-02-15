import gc
import psutil
import spacy
import torch
import datetime
import sys
import os
import torch.optim as optim

from torch.utils.data import  DataLoader
from src.x_reporto.data_loader.custom_dataset import CustomDataset
from src.utils import plot_image
from src.x_reporto.models.x_reporto_factory import XReporto
from src.x_reporto.data_loader.tokenizer import Tokenizer
# Utils 
from src.utils import save_model
from transformers import GPT2Tokenizer

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
    def __init__(self,training_csv_path: str =training_csv_path,validation_csv_path:str = validation_csv_path,
                 model:XReporto = None):
        '''
        inputs:
            training_csv_path (str): the path to the training csv file
            validation_csv_path (str): the path to the validation csv file
            model Optional[XReporto]: the x_reporto model instance to be trained.If not provided, the model is loaded from a .pth file.
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

        # Move to device
        self.model.to(DEVICE)

        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= LEARNING_RATE)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULAR_STEP_SIZE, gamma=SCHEDULAR_GAMMA)

        # create dataset
        # TODO Change to transform_type train
        self.checkpoint="healx/gpt-2-pubmed-medium"
        self.tokenizer = Tokenizer(self.checkpoint)

        self.dataset_train = CustomDataset(dataset_path= training_csv_path, transform_type='val',tokenizer=self.tokenizer)
        self.dataset_val = CustomDataset(dataset_path= validation_csv_path, transform_type='val',tokenizer=self.tokenizer)
        print("Dataset Loaded")
        
        # create data loader
        # TODO suffle Training Loaders
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        self.data_loader_val = DataLoader(dataset=self.dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print("DataLoader Loaded")
        # initialize the best loss to a large value
        self.best_loss = float('inf')
        # self.best_loss = 0.3904
        self.eval_best_loss = float('inf')

    def train(self):
        '''
        Train X-Reporto on the training dataset depending on the MODEL_STAGE.
        '''
        # make model in training mode
        print("Training Started")
        self.model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(self.data_loader_train):                
                
                images=object_detector_batch['image']
                # Move images to Device
                images = torch.stack([image.to(DEVICE) for image in images])
                length=len(images)

                # Moving Object Detector Targets to Device
                object_detector_targets=[]
                for i in range(len(images)):
                    new_dict={}
                    new_dict['boxes']=object_detector_batch['bboxes'][i].to(DEVICE)
                    new_dict['labels']=object_detector_batch['bbox_labels'][i].to(DEVICE)
                    object_detector_targets.append(new_dict)   

                selection_classifier_targets=None
                abnormal_classifier_targets=None
                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Selection Classifier
                    # Moving Selection Classifier Targets to Device
                    selection_classifier_targets=[]
                    for i in range(length):
                        phrase_exist=selection_classifier_batch['bbox_phrase_exists'][i]
                        selection_classifier_targets.append(phrase_exist)
                    selection_classifier_targets=torch.stack(selection_classifier_targets).to(DEVICE)
                    
                    # Abnormal Classifier
                    # Moving Object Detector Targets to Device
                    abnormal_classifier_targets=[]
                    for i in range(length):
                        bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal'][i]
                        abnormal_classifier_targets.append(bbox_is_abnormal)
                    abnormal_classifier_targets=torch.stack(abnormal_classifier_targets).to(DEVICE)

                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    # Language Model
                    # Moving Language Model Targets to Device
                    LM_targets=[]
                    input_ids=[]
                    attention_mask=[]
                    for i in range(len(images)):
                        phrase=LM_batch['label_ids'][i]
                        LM_targets.append(phrase)
                        input_ids.append(LM_batch['input_ids'][i])
                        attention_mask.append(LM_batch['attention_mask'][i])
                    LM_targets=torch.stack(LM_targets).to(DEVICE)
                    input_ids=torch.stack(input_ids).to(DEVICE)
                    attention_mask=torch.stack(attention_mask).to(DEVICE)
                    
                    loopLength= input_ids.shape[1]
                    for batch in range(BATCH_SIZE):
                        total_LM_losses=0
                        for i in range(0,loopLength,LM_Batch_Size):
                            # zero the parameter gradients
                            self.optimizer.zero_grad()

                            # Forward Pass
                            object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,stop= self.model(images,input_ids,attention_mask, object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_targets,batch,i)

                            if stop:
                                break
                            # Backward pass
                            # if(batch==0):
                            Total_loss=None
                            object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                            Total_loss=object_detector_losses_summation.clone()
                            Total_loss+=selection_classifier_losses
                            Total_loss+=abnormal_binary_classifier_losses

                            Total_loss+=LM_losses
                            total_LM_losses+=LM_losses
                            Total_loss.backward()

                            epoch_loss += Total_loss

                            # update the parameters
                            self.optimizer.step()
                        if DEBUG :
                          print(f'epoch.....: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} LM_losses: {total_LM_losses:.4f} total_Loss: {object_detector_losses_summation+selection_classifier_losses+abnormal_binary_classifier_losses+total_LM_losses:.4f}')
        
                else:
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward Pass
                    object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses= self.model(images,None,None, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets,None)
                    
                    # Backward pass
                    Total_loss=None
                    object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                    Total_loss=object_detector_losses_summation.clone()
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                        Total_loss+=selection_classifier_losses
                        Total_loss+=abnormal_binary_classifier_losses
        
                    Total_loss.backward()

                    epoch_loss += Total_loss

                    # update the parameters
                    self.optimizer.step()

                    if DEBUG :
                        print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f}  total_Loss: {Total_loss:.4f}')
                            
                # Free GPU memory
                del LM_losses
                del Total_loss
                del object_detector_losses
                del selection_classifier_losses
                del abnormal_binary_classifier_losses
                torch.cuda.empty_cache()
                gc.collect()
                print("losses deleted")

            # save the best model
            if(epoch_loss<self.best_loss and epoch%10==0) :
                self.best_loss=epoch_loss
                if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                    if TRAIN_RPN:
                        # Saving object_detector marked as rpn
                        print("Saving object_detector [Trained RPN]....")
                        save_model(model=self.model.object_detector,name="object_detector_rpn")
                    else:
                        # Saving Object Detector
                        print("Saving object_detector....")
                        save_model(model=self.model.object_detector,name="object_detector")
        
                elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                    # Saving Object Detector
                    print("Saving object_detector....")
                    save_model(model=self.model.object_detector,name="object_detector")

                    # Save Region Selection Classifier
                    print("Saving region_classifier....")
                    save_model(model=self.model.binary_classifier_selection_region,name="region_classifier")

                    # Save Abnormal Classifier
                    print("Saving abnormal_classifier....")
                    save_model(model=self.model.binary_classifier_region_abnormal,name='abnormal_classifier')

                elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    # Saving Object Detector
                    print("Saving object_detector....")
                    save_model(model=self.model.object_detector,name="object_detector")

                    # Save Region Selection Classifier
                    print("Saving region_classifier....")
                    save_model(model=self.model.binary_classifier_selection_region,name="region_classifier")

                    # Save Abnormal Classifier
                    print("Saving abnormal_classifier....")
                    save_model(model=self.model.binary_classifier_region_abnormal,name='abnormal_classifier')
   
                    # # Save LM
                    save_model(model=self.model.language_model,name='LM')
                                    

                # Logging the loss to a file
                with open("logs/loss.txt", "a") as myfile:
                    myfile.write(f'epoch: {epoch+1}/{EPOCHS}, epoch loss: {epoch_loss/len(self.data_loader_train):.4f}')
                    myfile.write("\n")
                # print the epoch loss
                print("\n")
                print(f'epoch: {epoch+1}/{EPOCHS}, epoch loss: {epoch_loss/len(self.data_loader_train):.4f}')
                print("\n")
            # update the learning rate
            # self.lr_scheduler.step()
                

    def Validate(self):
        '''
        Evaluate the X-Reporto model on the validation dataset
        '''
        # make model in training mode
        self.model.eval()
        with torch.no_grad():
        
            for epoch in range(EPOCHS):
                epoch_loss=0
                for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(self.data_loader_val):
                  
                    images=object_detector_batch['image']

                    # Move images to Device
                    images = torch.stack([image.to(DEVICE) for image in images])

                    # Moving Object Detector Targets to Device
                    object_detector_targets=[]
                    for i in range(len(images)):
                        new_dict={}
                        new_dict['boxes']=object_detector_batch['bboxes'][i].to(DEVICE)
                        new_dict['labels']=object_detector_batch['bbox_labels'][i].to(DEVICE)
                        object_detector_targets.append(new_dict)
                        
                    selection_classifier_targets=None
                    abnormal_classifier_targets=None
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                        # Selection Classifier
                        # Moving Selection Classifier Targets to Device
                        selection_classifier_targets=[]
                        for i in range(len(images)):
                            phrase_exist=selection_classifier_batch['bbox_phrase_exists'][i]
                            selection_classifier_targets.append(phrase_exist)
                        selection_classifier_targets=torch.stack(selection_classifier_targets).to(DEVICE)

                        # Abnormal Classifier
                        # Moving Object Detector Targets to Device
                        abnormal_classifier_targets=[]
                        for i in range(len(images)):
                            bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal'][i]
                            abnormal_classifier_targets.append(bbox_is_abnormal)
                        abnormal_classifier_targets=torch.stack(abnormal_classifier_targets).to(DEVICE)
                    if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                        # Language Model
                        # Moving Language Model Targets to Device
                        LM_targets=[]
                        input_ids=[]
                        attention_mask=[]
                        for i in range(len(images)):
                            phrase=LM_batch['label_ids'][i]
                            LM_targets.append(phrase)
                            input_ids.append(LM_batch['input_ids'][i])
                            attention_mask.append(LM_batch['attention_mask'][i])
                        LM_targets=torch.stack(LM_targets).to(DEVICE)
                        input_ids=torch.stack(input_ids).to(DEVICE)
                        attention_mask=torch.stack(attention_mask).to(DEVICE)
                        
                        loopLength= input_ids.shape[1]
                        for batch in range(BATCH_SIZE):
                            total_LM_losses=0
                            for i in range(0,loopLength,LM_Batch_Size):

                                # Forward Pass  
                                object_detector_losses,_,_,selection_classifier_losses,_,abnormal_binary_classifier_losses,_,LM_losses,_,stop= self.model(images,input_ids,attention_mask, object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_targets,batch,i,i+LM_Batch_Size>=loopLength and batch+1==BATCH_SIZE)
                                
                                if stop:
                                    break
                                Total_loss=None
                                object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                                Total_loss=object_detector_losses_summation.clone()
                                Total_loss+=selection_classifier_losses
                                Total_loss+=abnormal_binary_classifier_losses
                                Total_loss+=LM_losses
                                total_LM_losses+=LM_losses
                                epoch_loss += Total_loss

                            if DEBUG :
                                print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} LM_losses: {total_LM_losses:.4f} total_Loss: {object_detector_losses_summation+selection_classifier_losses+abnormal_binary_classifier_losses+total_LM_losses:.4f}')
                
                    else:
                        # Forward Pass
                        object_detector_losses,_,_,selection_classifier_losses,_,abnormal_binary_classifier_losses,_,LM_losses= self.model(images,None,None, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets,None,None,True)
                        
                        # Backward pass
                        Total_loss=None
                        object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                        Total_loss=object_detector_losses_summation.clone()
                        if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                            Total_loss+=selection_classifier_losses
                            Total_loss+=abnormal_binary_classifier_losses
            
                        epoch_loss += Total_loss

                        if DEBUG :
                            print(f'epoch: {epoch+1}, Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f}  total_Loss: {Total_loss:.4f}')

                    Total_loss=None
                    object_detector_losses_summation = object_detector_losses
                    Total_loss=object_detector_losses
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                        Total_loss+=selection_classifier_losses
                        Total_loss+=abnormal_binary_classifier_losses
                    if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                        Total_loss+=LM_losses

                    # Free GPU memory
                    Total_loss=Total_loss.to('cpu')
                    # move object_detector_losses to cpu
                    for key in object_detector_losses:
                        object_detector_losses[key]=object_detector_losses[key].to('cpu')
                    selection_classifier_losses=selection_classifier_losses.to('cpu')
                    abnormal_binary_classifier_losses=abnormal_binary_classifier_losses.to('cpu')
                    LM_losses=LM_losses.to('cpu')
                    del LM_losses
                    del Total_loss
                    del object_detector_losses
                    del selection_classifier_losses
                    del abnormal_binary_classifier_losses
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("losses deleted")
   
    def predict_and_display(self,predict_path_csv=None):
        '''
        Predict the output and display it with golden output.
        Each image is displayed in 5 sub-images with 6 labels in each sub-image.
        The golden output is dashed, and the predicted output is solid.

        Args:
            predict_path_csv (str): Path to the CSV file for prediction. (Default: None)
        '''
        if predict_path_csv==None:
                predicted_dataloader=self.data_loader_val
        else:
                predicted_data = CustomDataset(dataset_path= predict_path_csv, transform_type='val')

                # create data loader
                predicted_dataloader = DataLoader(dataset=predicted_data, batch_size=1, shuffle=False, num_workers=4)
                
        # make model in Evaluation mode
        self.model.eval()
        with torch.no_grad():
            epoch_loss=0
            for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(predicted_dataloader):
                # Check GPU memory usage
                
                images=object_detector_batch['image']

                # Move images to Device
                images = torch.stack([image.to(DEVICE) for image in images])

                # Moving Object Detector Targets to Device
                object_detector_targets=[]
                for i in range(len(images)):
                    new_dict={}
                    new_dict['boxes']=object_detector_batch['bboxes'][i].to(DEVICE)
                    new_dict['labels']=object_detector_batch['bbox_labels'][i].to(DEVICE)
                    object_detector_targets.append(new_dict)
                    
                selection_classifier_targets=None
                abnormal_classifier_targets=None
                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    # Selection Classifier
                    # Moving Selection Classifier Targets to Device
                    selection_classifier_targets=[]
                    for i in range(len(images)):
                        phrase_exist=selection_classifier_batch['bbox_phrase_exists'][i]
                        selection_classifier_targets.append(phrase_exist)
                    selection_classifier_targets=torch.stack(selection_classifier_targets).to(DEVICE)

                    # Abnormal Classifier
                    # Moving Object Detector Targets to Device
                    abnormal_classifier_targets=[]
                    for i in range(len(images)):
                        bbox_is_abnormal=abnormal_classifier_batch['bbox_is_abnormal'][i]
                        abnormal_classifier_targets.append(bbox_is_abnormal)
                    abnormal_classifier_targets=torch.stack(abnormal_classifier_targets).to(DEVICE)
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    # Language Model
                    # Moving Language Model Targets to Device
                    LM_targets=[]
                    input_ids=[]
                    attention_mask=[]
                    for i in range(len(images)):
                        phrase=LM_batch['label_ids'][i]
                        LM_targets.append(phrase)
                        input_ids.append(LM_batch['input_ids'][i])
                        attention_mask.append(LM_batch['attention_mask'][i])
                    LM_targets=torch.stack(LM_targets).to(DEVICE)
                    input_ids=torch.stack(input_ids).to(DEVICE)
                    attention_mask=torch.stack(attention_mask).to(DEVICE)
                    loopLength= input_ids.shape[1]
                    for batch in range(BATCH_SIZE):
                        total_LM_losses=0
                        for i in range(0,loopLength,LM_Batch_Size):
                            # Forward Pass
                            object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions,LM_losses,LM_predict,stop= self.model(images,input_ids,attention_mask, object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_targets,batch,i,i+LM_Batch_Size>=loopLength-1)
                            if stop:
                                break
                            print("selection looses ",selection_classifier_losses)
                            object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                            Total_loss=object_detector_losses_summation.clone()
                            Total_loss+=selection_classifier_losses
                            Total_loss+=abnormal_binary_classifier_losses
                            Total_loss+=LM_losses
                            total_LM_losses+=LM_losses
                            print(LM_predict)
                            generated_sents_for_selected_regions = self.tokenizer.batch_decode(LM_predict, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            print("Generated Sentences for Selected Regions: ", generated_sents_for_selected_regions)
                            # sys.exit()
                            with open("logs/predictions.txt", "a") as myfile:
                                # don't know the text TODO: 
                                # myfile.write
                                myfile.write("\n")

                            if DEBUG :
                                print(f'Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f} LM_losses: {total_LM_losses:.4f} total_Loss: {object_detector_losses_summation+selection_classifier_losses+abnormal_binary_classifier_losses+total_LM_losses:.4f}')
                
                else:
                    # Forward Pass
                    object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions,LM_losses= self.model(images,None,None, object_detector_targets ,selection_classifier_targets,abnormal_classifier_targets,None,None,True)
                    Total_loss=None
                    object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                    Total_loss=object_detector_losses_summation.clone()
                    if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                        Total_loss+=selection_classifier_losses
                        Total_loss+=abnormal_binary_classifier_losses
                    epoch_loss += Total_loss

                    if DEBUG :
                        print(f' Batch {batch_idx + 1}/{len(self.data_loader_train)} object_detector_Loss: {object_detector_losses_summation:.4f} selection_classifier_Loss: {selection_classifier_losses:.4f} abnormal_classifier_Loss: {abnormal_binary_classifier_losses:.4f}  total_Loss: {Total_loss:.4f}')
                
                plot_image(images[0].cpu(),object_detector_targets[0]["labels"].tolist(),object_detector_targets[0]["boxes"].tolist(),object_detector_detected_classes[0].tolist(),object_detector_boxes[0].tolist())

                Total_loss=None
                object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                Total_loss=object_detector_losses_summation.clone()
                if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                    Total_loss+=selection_classifier_losses
                    Total_loss+=abnormal_binary_classifier_losses
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    Total_loss+=LM_losses
                
                
                Total_loss=None
                object_detector_losses_summation = sum(loss for loss in object_detector_losses.values())
                Total_loss=object_detector_losses_summation.clone()
                if MODEL_STAGE==ModelStage.CLASSIFIER.value:
                    Total_loss+=selection_classifier_losses
                    Total_loss+=abnormal_binary_classifier_losses
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                    Total_loss+=LM_losses

                if DEBUG :
                    # FreeGPU gpu memory
                    del Total_loss
                    del object_detector_losses
                    del selection_classifier_losses
                    del abnormal_binary_classifier_losses
                    torch.cuda.empty_cache()
                    
                    # To Test Overfitting break
                    break


    def generate_sentences(self,predict_path_csv=None):
        '''
        Predict the output and display it with golden output.
        Each image is displayed in 5 sub-images with 6 labels in each sub-image.
        The golden output is dashed, and the predicted output is solid.

        Args:
            predict_path_csv (str): Path to the CSV file for prediction. (Default: None)
        '''
        if predict_path_csv==None:
                predicted_dataloader=self.data_loader_val
        else:
                predicted_data = CustomDataset(dataset_path= predict_path_csv, transform_type='val',tokenizer=self.tokenizer)

                # create data loader
                predicted_dataloader = DataLoader(dataset=predicted_data, batch_size=1, shuffle=False, num_workers=4)
                
        # make model in Evaluation mode
        self.model.eval()
        with torch.no_grad():
            epoch_loss=0
            for batch_idx,(object_detector_batch,selection_classifier_batch,abnormal_classifier_batch,LM_batch) in enumerate(predicted_dataloader):
                # Check GPU memory usage
                images=object_detector_batch['image']
                reference_sentences=[]
                for i in range(len(images)):
                       reference_sentences.append(LM_batch['bbox_phrase'][i])
                       
                # Move images to Device
                images = torch.stack([image.to(DEVICE) for image in images])
                loopLength=29
                for batch in range(BATCH_SIZE):
                    for i in range(0,loopLength,LM_Batch_Size):
                        # Forward Pass
                        # LM_sentances,stop= self.model(images=images, object_detector_targets=object_detector_targets,selection_classifier_targets=selection_classifier_targets,batch=batch,index=i,delete=i+LM_Batch_Size>=loopLength-1,generate_sentence=True)
                        LM_sentances,stop= self.model(images=images,batch=batch,index=i,delete=i+LM_Batch_Size>=loopLength-1,generate_sentence=True)
                        tokenizer = GPT2Tokenizer.from_pretrained("healx/gpt-2-pubmed-medium")
                        for sentence in LM_sentances:
                            generated_sentence_for_selected_regions = tokenizer.decode(sentence.tolist(),skip_special_tokens=True)
                            print("generated_sents_for_selected_regions",generated_sentence_for_selected_regions)
                        with open("logs/predictions.txt", "a") as myfile:
                            myfile.write(generated_sentence_for_selected_regions)
                            myfile.write("\n")
                        if stop:
                            break
                        
                
    def save_check_point(self,epoch):
        checkpoint={
            "epoch":epoch,
            "optim_state":self.optimizer.state_dict()
        }


        if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
            checkpoint['object_detector']=self.model.object_detector.state_dict()

                
        elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
                checkpoint['object_detector']=self.model.object_detector.state_dict()            

                checkpoint['region_classifier']=self.model.abnormal_classifier.state_dict()
                checkpoint['abnormal_classifier']=self.model.abnormal_classifier.state_dict()

                
        elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
                checkpoint['object_detector']=self.model.object_detector.state_dict()            
            
                checkpoint['region_classifier']=self.model.abnormal_classifier.state_dict()
                checkpoint['abnormal_classifier']=self.model.abnormal_classifier.state_dict()

                # Save Language Model

        # Get the current date and time
        current_datetime = datetime.datetime.now()
        # Format the date and time to be part of the filename
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        # Create the filename with the formatted datetime
        name = f"ckpt_{formatted_datetime}"

        # Save Checkpoint File
        torch.save(checkpoint,"models/" + RUN + '/checkpoints/' + name + ".pth")
    
    def load_check_point(self,epoch,name=None):
        directory_path="models/" + RUN + '/checkpoints/'

        if name is None:
            # Load latest check point
            all_files = os.listdir(directory_path)

            # Filter out non-pth files
            pth_files = [file for file in all_files if file.endswith(".pth")]

            # Sort the files based on creation time
            sorted_files = sorted(pth_files, key=lambda x: os.path.getctime(os.path.join(directory_path, x)), reverse=True)

            # Get the latest file
            latest_file = sorted_files[0] if sorted_files else None

            checkpoint_path=directory_path+latest_file

        else:
            # Load Specified
            checkpoint_path=directory_path + name + ".pth"
        
        checkpoint=torch.load(checkpoint_path)
        
        epoch=checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optim_state'])


        if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
            checkpoint['object_detector']=self.model.object_detector.load_state_dict(checkpoint['object_detector'])
                
        elif MODEL_STAGE==ModelStage.CLASSIFIER.value:
            checkpoint['object_detector']=self.model.object_detector.load_state_dict(checkpoint['object_detector'])

            checkpoint['region_classifier']=self.model.abnormal_classifier.load_state_dict(checkpoint['region_classifier'])
            checkpoint['abnormal_classifier']=self.model.abnormal_classifier.load_state_dict(checkpoint['abnormal_classifier'])

                
        elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
            checkpoint['object_detector']=self.model.object_detector.load_state_dict(checkpoint['object_detector'])

            checkpoint['region_classifier']=self.model.abnormal_classifier.load_state_dict(checkpoint['region_classifier'])
            checkpoint['abnormal_classifier']=self.model.abnormal_classifier.load_state_dict(checkpoint['abnormal_classifier'])

            # Load Language Model ckpt


# def set_data(args):
#     # read hyper-parameters from terminal
#     # if not set read from config.py file
#     if (len(args)>1):
#         global EPOCHS
#         EPOCHS = int(args[1])
#         if (len(args)>2):
#             global LEARNING_RATE
#             LEARNING_RATE=float(args[2])
#             if (len(args)>3):
#                 global BATCH_SIZE
#                 BATCH_SIZE=int(args[3])
#                 if (len(args)>4):
#                     global MODEL_STAGE
#                     MODEL_STAGE=int(args[4])
#                     if (len(args)>5):
#                         global SCHEDULAR_STEP_SIZE
#                         SCHEDULAR_STEP_SIZE=float(args[5])
#                         if (len(args)>6):
#                             global SCHEDULAR_GAMMA
#                             SCHEDULAR_GAMMA=float(args[6])
        
# import argparse
if __name__ == '__main__':
    
    print("Using Configuration:")
    print("Model Stage", MODEL_STAGE)
    print("Device", DEVICE)

    print("Epochs:", EPOCHS)
    print("Learning Rate:", LEARNING_RATE)
    print("Batch Size:", BATCH_SIZE)
    print("Scheduler Step Size:", SCHEDULAR_STEP_SIZE)
    print("Scheduler Gamma:", SCHEDULAR_GAMMA)
    print("Debug",DEBUG)

    print("Continue train",CONTINUE_TRAIN)
    print("Train RPN",TRAIN_RPN)
    print("Run",RUN)

    print("Abnormal Region Pos Weight:", ABNORMAL_CLASSIFIER_POS_WEIGHT)
    print("Region Selection Pos Weight:", REGION_SELECTION_CLASSIFIER_POS_WEIGHT)

    # create run folder
    folder_path="models/" + str(RUN)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")
    x_reporto_model = XReporto().create_model()

    # Create an XReportoTrainer instance with the X-Reporto model
    trainer = XReportoTrainer(model=x_reporto_model)

    # Alternatively, create an XReportoTrainer instance without specifying the model
    # trainer = XReportoTrainer()

    # Train the X-Reporto model on the training dataset
    # trainer.train()

    # # Run Validation
    # trainer.Validate()

    # # Predict and display results
    # trainer.predict_and_display(predict_path_csv='datasets/train.csv')
    trainer.generate_sentences(predict_path_csv='datasets/train.csv')