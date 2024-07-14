import gc
import sys
import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional, List, Dict

from config import *
# Utils 
from src.utils import load_model
import numpy as np
from src.denoiser.utils import save2image


# Modules
from src.object_detector.models.object_detector_factory import ObjectDetector
from src.binary_classifier.models.binary_classifier_selection_region_factory import BinaryClassifierSelectionRegion
from src.binary_classifier.models.binary_classifier_region_abnormal_factory import BinaryClassifierRegionAbnormal
from src.language_model.GPT2.gpt2_model import CustomGPT2
from src.language_model.GPT2.config import Config
from transformers import GPT2Tokenizer
from  src.x_reporto.data_loader.custom_augmentation import CustomAugmentation
from src.denoiser.models.gan_model import TomoGAN
from src.denoiser.options.test_options import TestOptions
import logging

from albumentations.pytorch import ToTensorV2
import albumentations as A

transform = A.Compose([
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2()
            ])

class XReportoV1(nn.Module):
    """
    A modular model for object detection and binary classification.

    Attributes:
        - num_classes (int): The number of classes for object detection.

    Modules:
        - object_detector (ObjectDetectorWrapper): Object detector module.
        - binary_classifier_selection_region (BinaryClassifierSelectionRegionWrapper): Binary classifier for region selection.
        - binary_classifier_region_abnormal (BinaryClassifierRegionAbnormalWrapper): Binary classifier for abnormal region detection.
        - language_model (CustomGPT2): Language model for generating reports.
    """

    def __init__(self,object_detector_path=None,region_classifier_path=None,abnormal_classifier_path=None,language_model_path=None):
        super().__init__()
        self.num_classes=30

        self.object_detector = ObjectDetector().create_model()

        self.arg= self.arg=TestOptions()
        self.denoiser = TomoGAN(self.arg)

        if OPERATION_MODE==OperationMode.INFERENCE.value or MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
            self.binary_classifier_selection_region = BinaryClassifierSelectionRegion().create_model()
            self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal().create_model()
        if OPERATION_MODE!=OperationMode.INFERENCE.value:
            self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal().create_model()
        
        if OPERATION_MODE==OperationMode.INFERENCE.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
            config = Config()
            # load small gpt2 config
            config.d_model = 768
            config.d_ff1 = 768
            config.d_ff2 = 768
            config.d_ff3 = 768
            config.num_heads = 12
            config.num_layers = 12
            config.vocab_size = 50257
            config.pretrained_model = "gpt2"
            config.max_seq_len = 1024
            config.ignore_index = -100
            config.use_checkpointing = False
            image_config = Config()
            image_config.d_model = 1024
            image_config.d_ff1 = 1024
            image_config.d_ff2 = 1024
            # image_config.d_ff2 = 768
            image_config.d_ff3 = 768
            image_config.num_heads = 16
            image_config.num_layers = 24
            image_config.vocab_size = 50257
            self.language_model = CustomGPT2(config,image_config)
            # convert the model to half precision
            # self.language_model.half()
            # self.language_model.convert_to_half()

        if OPERATION_MODE==OperationMode.INFERENCE.value:
            # Load the models
            self.denoiser.load_models()
            self.object_detector.load_state_dict(torch.load(object_detector_path))
            self.binary_classifier_selection_region.load_state_dict(torch.load(region_classifier_path))
            self.binary_classifier_region_abnormal.load_state_dict(torch.load(abnormal_classifier_path))
            self.language_model.load_state_dict(torch.load(language_model_path))

        elif OPERATION_MODE==OperationMode.TRAINING.value:
            if CONTINUE_TRAIN:
                if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value and TRAIN_RPN:
                        
                    logging.info("Loading object_detector [Trained RPN]....")
                    print("Loading object_detector [Trained RPN]....")
                    load_model(model=self.object_detector,name='object_detector_rpn_best')
                elif MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value and TRAIN_ROI:
                    logging.info("Loading object_detector [Trained ROI]....")
                    print("Loading object_detector [Trained ROI]....")
                    load_model(model=self.object_detector,name='object_detector_roi_best')
                else:
                    # Load full object detector
                    logging.info("Loading object_detector .....")
                    print("Loading object_detector .....")
                    load_model(model=self.object_detector,name='object_detector_best')
                    
                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load the Region Selection Classifier to continue training
                    logging.info("Loading region_classifier .....")
                    print("Loading region_classifier .....")
                    load_model(model=self.binary_classifier_selection_region,name='region_classifier_best')

                    # Load the Abnormal Classifier to continue training
                    logging.info("Loading abnormal_classifier .....")
                    print("Loading abnormal_classifier .....")
                    load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier_best')

                    if FREEZE_OBJECT_DETECTOR or FREEZE:
                        # Freezing Object Detector Model [including Backbone, RPN, RoI Heads]
                        for param in self.object_detector.object_detector.parameters():
                            param.requires_grad = False
                        logging.info("All object_detector paramets are Frozen")   
                        print("All object_detector paramets are Frozen")      
                    else: logging.info("All object_detector paramets are Trainable")      
    
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load Language Model to continue training
                    logging.info("Loading language_model .....")
                    print("Loading language_model .....")
                    load_model(model=self.language_model,name='LM_best')
                    if FREEZE:
                        # Freezing Selection Region Binary Classifier
                        for param in self.binary_classifier_selection_region.selection_binary_classifier.parameters():
                            param.requires_grad = False

                        # Freezing Abnormal Region Binary Classifier
                        for param in self.binary_classifier_region_abnormal.abnormal_binary_classifier.parameters():
                            param.requires_grad = False   
            else:
                if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                    if TRAIN_RPN:
                        pass
                    elif TRAIN_ROI:
                        logging.info("Loading object_detector [Trained RPN]....")
                        print("Loading object_detector [Trained RPN]....")
                        load_model(model=self.object_detector,name='object_detector_rpn_best')
                    else:
                        logging.info("Loading object_detector [Trained RPN]....")
                        print("Loading object_detector [Trained RPN]....")
                        load_model(model=self.object_detector,name='object_detector_roi_best')
                    

                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load the object_detector to continue training
                    logging.info("Loading object_detector .....")
                    print("Loading object_detector .....")
                    load_model(model=self.object_detector,name='object_detector_best')

                    if FREEZE_OBJECT_DETECTOR or FREEZE:
                        # Freezing Object Detector Model [including Backbone, RPN, RoI Heads]
                        for param in self.object_detector.object_detector.parameters():
                            param.requires_grad = False
                        logging.info("All object_detector paramets are Frozen")  
                        print("All object_detector paramets are Frozen")       
                    else: logging.info("All object_detector paramets are Trainable")         


                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load the Region Selection Classifier to start training
                    logging.info("Loading region_classifier .....")
                    print("Loading region_classifier .....")
                    load_model(model=self.binary_classifier_selection_region,name='region_classifier_best')
                    if FREEZE:
                        # Freezing Selection Region Binary Classifier
                        for param in self.binary_classifier_selection_region.selection_binary_classifier.parameters():
                            param.requires_grad = False

                    # Load the Abnormal Classifier to start training
                    logging.info("Loading abnormal_classifier .....")
                    print("Loading abnormal_classifier .....")
                    load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier_best')
                    if FREEZE:
                        # Freezing Abnormal Region Binary Classifier
                        for param in self.binary_classifier_region_abnormal.abnormal_binary_classifier.parameters():
                            param.requires_grad = False
                            # if  GENERATE_REPORT:
                            #     load_model(model=self.language_model,name='LM_best')

        elif OPERATION_MODE==OperationMode.VALIDATION.value or OPERATION_MODE==OperationMode.EVALUATION.value:
            self.denoiser.load_models()
            if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                logging.info("Loading object_detector .....")
                print("Loading object_detector .....")
                load_model(model=self.object_detector,name='object_detector_best')

            if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                logging.info("Loading object_detector .....")
                print("Loading object_detector .....")
                load_model(model=self.object_detector,name='object_detector_best')
                logging.info("Loading region_classifier .....")
                print("Loading region_classifier .....")
                load_model(model=self.binary_classifier_selection_region,name='region_classifier_best')
                logging.info("Loading abnormal_classifier .....")
                print("Loading abnormal_classifier .....")
                load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier_best')

            if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                logging.info("Loading language_model .....")
                print("Loading language_model .....")
                load_model(model=self.language_model,name='LM_best')         


            

    def forward(self,images: Tensor ,input_ids=None,attention_mask=None, object_detector_targets: Optional[List[Dict[str, Tensor]]] = None, selection_classifier_targets: Tensor=None,abnormal_classifier_targets: Tensor = None,language_model_targets: Tensor= None,batch:Optional[int]=None,index:Optional[int]=None,delete:Optional[bool]= False,generate_sentence :Optional[bool]=False,use_beam_search :Optional[bool] = False,validate_during_training:Optional[bool]=False,
                selected_models: List[bool] = [True,True,True,True]):
        '''
        Forward pass through the X-ReportoV1 model.

        Args:
            - images (Tensor): Input images of shape [batch_size x 1 x 512 x 512] (grey-scaled images) [Normalized 0-1].
            - object_detector_targets (Optional[List[Dict[str, Tensor]]]): List of dictionaries containing bounding box targets for each batch element
                Each dictionary in the list should have the following keys:
                - 'boxes' (FloatTensor[N, 4]): Ground-truth boxes in [x1, y1, x2, y2] format, where N is the number of bounding boxes detected.
                - 'labels' (Int64Tensor[N]): Ground-truth labels for each box, where N is the number of bounding boxes detected.
                If None, the model assumes inference mode without ground truth labels.
            - selection_classifier_targets (Optional[Tensor]):Binary Tensor of shape [batch_size x,29]
                Ground truth indicating whether a phrase exists in the region or not.
                In CLASSIFIER Mode:
                    If None , the model assumes inference mode without ground truth labels.
            - abnormal_classifier_targets (Optional[Tensor]):Binary Tensor of shape [batch_size x,29]
                Ground truth indicating whether a region is abnormal or not.
                In CLASSIFIER Mode:
                    If None , the model assumes inference mode without ground truth labels.
            - language_model__targets
            -batch (Optional[int]): The batch index.
            -index (Optional[int]): The index of the input in the batch.
            -delete (Optional[bool]): If True, delete the input tensors from the GPU memory after the forward pass.
            -generate_sentence (Optional[bool]): If True, generate a sentence using the language model.
            -use_beam_search (Optional[bool]): If True, use beam search to generate a sentence.

        Returns:
            If in OBJECT_DETECTOR Mode:
                If in training mode:
                 - object_detector_losses Dict: Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                - selection_classifier_losses: 0
                - abnormal_binary_classifier_losses: 0
                - LM_output: 0

                If in Validation mode:
                - object_detector_losses Dict: Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                    Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                    Indicates if the object detector has detected the region/class or not.

                If in Inference mode:           
                    - object_detector_losses (Dict): Empty Dictionary
                    - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                        Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                    - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                        Indicates if the object detector has detected the region/class or not.


            If in CLASSIFIER Mode:
                If in training mode:
                    - object_detector_losses (Dict): Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                   - selection_classifier_losses (Tensor): Loss of the Selection Region Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                   - abnormal_binary_classifier_losses (Tensor): Loss of the Abnormal Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - LM_output: 0

                If in Validation mode:
                    - object_detector_losses (Dict): Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                    - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                        Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                    - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                        Indicates if the object detector has detected the region/class or not.

                    - selection_classifier_losses(Tensor): Loss of the Selection Region Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - selected_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                        Indicating selected regions

                    - abnormal_binary_classifier_losses(Tensor): Loss of the Abnormal Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                       Indicating predicted abnormal regions.

                If in Inference mode:
                    - object_detector_losses (Dict): Empty Dictionary
                    - object_detector_boxes(Tensor): Tensor of shape [batch_size x 29 x 4]
                        Detected bounding boxes in [x1, y1, x2, y2] format for each of the 29 regions.
                    - object_detector_detected_classes (Tensor): Boolean tensor of shape [batch_size x 29].
                        Indicates if the object detector has detected the region/class or not.

                    - selection_classifier_losses: None.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - selected_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                        Indicating selected regions

                    - abnormal_binary_classifier_losses: None.
                    - predicted_abnormal_regions(Tensor): Boolean Tensor of shape [batch_size x 29] 
                        Indicating predicted abnormal regions.    
            if in LANGUAGE_MODEL Mode:
                If in training mode:
                    - object_detector_losses (Dict): Dictionary containing object detector losses with keys:
                        - 'loss_objectness' (Tensor): Objectness loss.
                        - 'loss_rpn_box_reg' (Tensor): RPN box regression loss.
                        - 'loss_classifier' (Tensor): Classifier loss.
                        - 'loss_box_reg' (Tensor): Box regression loss.
                    - selection_classifier_losses (Tensor): Loss of the Selection Region Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - abnormal_binary_classifier_losses (Tensor): Loss of the Abnormal Binary Classifier.
                        The format is similar to `tensor(1.0215, device='cuda:0', grad_fn=<....>)`.
                    - LM_output (Tuple): Tuple containing the language model output with keys:
                        - 'loss' (Tensor): Language model loss.
                        - 'logits' (Tensor): Logits of the language model.
                        - 'hidden_states' (Tensor): Hidden states of the language model.
                        - 'attentions' (Tensor): Attention weights of the language model.
                    - stop (bool): If True, the batch index has reached the end of the dataset.

       '''

        stop=False
        if OPERATION_MODE==OperationMode.INFERENCE.value:
            print("Inference Mode Forward Pass")
            sum =  np.sum(selected_models)
            if sum==0:
                raise ValueError("At least one model must be selected for inference")
            # Denoiser 
            self.denoiser.set_input(images)
            self.denoiser.forward()
            images = self.denoiser.fake_C
            images = (images - images.min()) / (images.max() -images.min())
            images=images.detach().cpu().numpy()
            images = images * 255
            images = images.astype(np.uint8)

            if sum == 1:
                return images        
            # Convert the batch to a list of individual images
            images_list = [images[i, 0, :, :] for i in range(images.shape[0])]

            # Apply the transformation to each image
            transformed_images_list = [transform(image=image)["image"] for image in images_list]

            # Convert the list back to a batch
            images = torch.stack(transformed_images_list)
            images=images.cuda()
            # Object Detector
            self.object_detector(images=images)
            _,bounding_boxes,detected_classes,object_detector_features = self.object_detector(images=images)


            #print("bounding_boxes",bounding_boxes.shape) #[batch_size x 29 x 4]
            #print("detected_classes",detected_classes.shape) #[batch_size x 29]
            #print("object_detector_features",object_detector_features.shape) # [batch_size x 29 x 1024]

            if sum == 2:
                    # squeeze the classes
                classes = torch.squeeze(detected_classes)
                boxes_labels = []
                for i in range(classes.shape[0]):
                    if classes[i]:
                        boxes_labels.append(i)

                bounding_boxes = torch.squeeze(bounding_boxes)
                del detected_classes
                del object_detector_features
                return bounding_boxes, boxes_labels
            # Abnormal Classifier
            _,abnormal_regions =self.binary_classifier_region_abnormal(object_detector_features,detected_classes)
            # Binary Classifier
            _,selected_regions,selected_region_features=self.binary_classifier_selection_region(object_detector_features,detected_classes)
            # print("selected_regions: ",selected_regions.shape)  #[batch_size x 29] 
            #print(selected_region_features.shape) #[num_regions_selected_in_batch,1024]

            # _,abnormal_regions=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            # print(abnormal_regions) # Boolean Tensor of shape [batch_size x 29] 


            # Language Model
            LM_sentences=[]
            object_detector_features = object_detector_features[selected_regions]
            for lm_index in range(0,len(object_detector_features),LM_Batch_Size):
                if use_beam_search:
                    LM_sentences_batch=self.language_model.beam_search(max_length=100,image_hidden_states=object_detector_features[lm_index:lm_index+LM_Batch_Size,:],beam_size =6,device=DEVICE,debug=False)
                else:
                    LM_sentences_batch=self.language_model.generate(max_length=100,image_hidden_states=object_detector_features[lm_index:lm_index+LM_Batch_Size,:],greedy=True,device=DEVICE)
                LM_sentences.extend(LM_sentences_batch)
            # print("LM_sentences",len(LM_sentences))# [num_regions_selected_in_batch,]

            # squeeze the classes
            # detected_classes = detected_classes[selected_regions]
            # classes = torch.squeeze(detected_classes)
            selected_regions_squeezed = torch.squeeze(selected_regions)
            boxes_labels = []
            for i in range(len(selected_regions_squeezed)):
                if selected_regions_squeezed[i]:
                    boxes_labels.append(i)

            del detected_classes
            del object_detector_features
            gc.collect()
            # return denoised images, bounding boxes, selected regions, abnormal regions, and generated sentences
            return images, bounding_boxes[selected_regions],selected_regions,abnormal_regions,  LM_sentences, boxes_labels
        
        elif OPERATION_MODE==OperationMode.TRAINING.value and self.training:
            # Training
            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
            
            if delete:
                # Free GPU memory 
                # images=images.to('cpu')
                # # move object_detector_targets to cpu
                # for i in range(len(object_detector_targets)):
                #     object_detector_targets[i]['boxes']=object_detector_targets[i]['boxes'].to('cpu')
                #     object_detector_targets[i]['labels']=object_detector_targets[i]['labels'].to('cpu')
                # delete boxes and labels in object_detector_targets
                # for i in range(len(object_detector_targets)):
                #     del object_detector_targets[i]['boxes']
                #     del object_detector_targets[i]['labels']
                del images
                del object_detector_targets
                del object_detector_boxes
                
                torch.cuda.empty_cache()

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,0,0,0
            # Stage(2) Binary Classifier
            object_detector_detected_classes=object_detector_detected_classes.to(DEVICE)
            selection_classifier_losses,_,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,_=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            
            if delete:
                # free gpu memory
                # abnormal_classifier_targets=abnormal_classifier_targets.to('cpu')
                # object_detector_detected_classes=object_detector_detected_classes.to('cpu')
                del abnormal_classifier_targets
                del object_detector_detected_classes
                torch.cuda.empty_cache()

            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,0
            # Stage(3) Language Model
            valid_input_ids, valid_attention_mask, valid_object_detector_features ,valid_labels= self.filter_inputs_to_language_model(selection_classifier_targets, input_ids, attention_mask, object_detector_features,language_model_targets)
            if delete or True:
                # selection_classifier_targets=selection_classifier_targets.to('cpu')
                del selection_classifier_targets
                del object_detector_features
                del input_ids
                del attention_mask
                del language_model_targets
                
                torch.cuda.empty_cache()
            if index>=len(valid_input_ids):
                # return losses of zeros
                object_detector_losses={'loss_objectness':torch.tensor(0.0,requires_grad=True).to(DEVICE),'loss_rpn_box_reg':torch.tensor(0.0,requires_grad=True).to(DEVICE),'loss_classifier':torch.tensor(0.0,requires_grad=True).to(DEVICE),'loss_box_reg':torch.tensor(0.0,requires_grad=True).to(DEVICE)}
                selection_classifier_losses=torch.tensor(0.0,requires_grad=True).to(DEVICE)
                abnormal_binary_classifier_losses=torch.tensor(0.0,requires_grad=True).to(DEVICE)
                LM_losses= torch.tensor(0.0,requires_grad=True).to(DEVICE)
                return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,True
            
            # if (index+LM_Batch_Size) >= len(valid_input_ids):
            #     stop=True
            # loop on all the valid_input_ids using LM_Batch_Size
            LM_losses=0
            for lm_index in range(0,len(valid_input_ids),LM_Batch_Size):
                LM_output=self.language_model(input_ids=valid_input_ids[lm_index:lm_index+LM_Batch_Size,:],image_hidden_states=valid_object_detector_features[lm_index:lm_index+LM_Batch_Size,:],attention_mask=valid_attention_mask[lm_index:lm_index+LM_Batch_Size,:],labels=valid_labels[lm_index:lm_index+LM_Batch_Size,:])
                LM_losses=LM_losses+LM_output[0]
                logging.debug(f"Current LM_losses: {LM_output[0]}")
                logging.debug(f"LM_losses: {LM_losses}")
            # LM_output=self.language_model(input_ids=valid_input_ids[index:index+LM_Batch_Size,:],image_hidden_states=valid_object_detector_features[index:index+LM_Batch_Size,:],attention_mask=valid_attention_mask[index:index+LM_Batch_Size,:],labels=valid_labels[index:index+LM_Batch_Size,:])
            
            if delete:

                del valid_input_ids
                del valid_attention_mask
                del valid_object_detector_features
                del valid_labels
                torch.cuda.empty_cache()
                gc.collect()
            return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,stop
        else:
            # Stage (0) Denoiser
            # Denoiser 
            self.denoiser.set_input(images)
            self.denoiser.forward()
            images = self.denoiser.fake_C
            images = (images - images.min()) / (images.max() -images.min())
            images=images.detach().cpu().numpy()
            images = images * 255
            images = images.astype(np.uint8)           
            # Convert the batch to a list of individual images
            images_list = [images[i, 0, :, :] for i in range(images.shape[0])]

            # Apply the transformation to each image
            transformed_images_list = [transform(image=image)["image"] for image in images_list]

            # Convert the list back to a batch
            images = torch.stack(transformed_images_list)
            images=images.cuda()


            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)

            if delete:
                # Free GPU memory 
                images=images.to('cpu')
                # move object_detector_targets to cpu
                # for i in range(len(object_detector_targets)):
                #     object_detector_targets[i]['boxes']=object_detector_targets[i]['boxes'].to('cpu')
                #     object_detector_targets[i]['labels']=object_detector_targets[i]['labels'].to('cpu')
                del images
                #TODO: should be removed ?
                # del object_detector_targets
                torch.cuda.empty_cache()
            
            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                if OPERATION_MODE==OperationMode.VALIDATION.value or validate_during_training:
                    return object_detector_losses,0,0,0
                elif OPERATION_MODE==OperationMode.EVALUATION.value or OPERATION_MODE==OperationMode.TESTING.value:
                    return object_detector_losses,object_detector_boxes,object_detector_detected_classes,0,None,0,None,None,None,None
            
            # Stage(2) Binary Classifier
            selection_classifier_losses,selected_regions,selected_region_features=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            if MODEL_STAGE == ModelStage.CLASSIFIER.value or MODEL_STAGE == ModelStage.LANGUAGE_MODEL.value :
                abnormal_binary_classifier_losses,predicted_abnormal_regions=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
             
            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                if delete:
                # free gpu memory
                  selection_classifier_targets=selection_classifier_targets.to('cpu')
                  del selection_classifier_targets
                  torch.cuda.empty_cache()
                if OPERATION_MODE==OperationMode.VALIDATION.value or validate_during_training: 
                    return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,0
                elif OPERATION_MODE==OperationMode.EVALUATION.value or OPERATION_MODE==OperationMode.TESTING.value:
                    return object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions,None,None,None
            
            # Stage(3) Language Model
            if OPERATION_MODE==OperationMode.VALIDATION.value or validate_during_training: 
                valid_input_ids, valid_attention_mask, valid_object_detector_features ,valid_labels= self.filter_inputs_to_language_model(selection_classifier_targets, input_ids, attention_mask, object_detector_features,language_model_targets)
                if delete or True:
                    selection_classifier_targets=selection_classifier_targets.to('cpu')
                    del selection_classifier_targets
                    torch.cuda.empty_cache()
                if index>=len(valid_input_ids):
                    # return losses of zeros
                    object_detector_losses={'loss_objectness':torch.tensor(0.0,requires_grad=True).to(DEVICE),'loss_rpn_box_reg':torch.tensor(0.0,requires_grad=True).to(DEVICE),'loss_classifier':torch.tensor(0.0,requires_grad=True).to(DEVICE),'loss_box_reg':torch.tensor(0.0,requires_grad=True).to(DEVICE)}
                    selection_classifier_losses=torch.tensor(0.0,requires_grad=True).to(DEVICE)
                    abnormal_binary_classifier_losses=torch.tensor(0.0,requires_grad=True).to(DEVICE)
                    LM_losses= torch.tensor(0.0,requires_grad=True).to(DEVICE)
                    return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,True
                LM_losses=0
                for lm_index in range(0,len(valid_input_ids),LM_Batch_Size):
                    LM_output=self.language_model(input_ids=valid_input_ids[index:index+LM_Batch_Size,:],image_hidden_states=valid_object_detector_features[index:index+LM_Batch_Size,:],attention_mask=valid_attention_mask[index:index+LM_Batch_Size,:],labels=valid_labels[index:index+LM_Batch_Size,:])
                    LM_losses=LM_losses+LM_output[0]

                if delete:
                    # Free GPU memory
                    object_detector_features=object_detector_features.to('cpu')
                    input_ids=input_ids.to('cpu')
                    attention_mask=attention_mask.to('cpu')
                    del object_detector_features
                    del input_ids
                    del attention_mask
                    del valid_input_ids
                    del valid_attention_mask
                    del valid_object_detector_features
                    torch.cuda.empty_cache()
                    gc.collect()
                    return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_losses,stop
                
            elif OPERATION_MODE==OperationMode.EVALUATION.value:
                # selected_regions=torch.ones_like(selected_regions)
                LM_sentences=[]
                object_detector_features = object_detector_features[selected_regions]
                for lm_index in range(0,len(object_detector_features),LM_Batch_Size):
                    if use_beam_search:
                        LM_sentences_batch=self.language_model.beam_search(max_length=150,image_hidden_states=object_detector_features[lm_index:lm_index+LM_Batch_Size,:],beam_size =8,device=DEVICE,debug=False)
                    else:
                        LM_sentences_batch=self.language_model.generate(max_length=150,image_hidden_states=object_detector_features[lm_index:lm_index+LM_Batch_Size,:],greedy=True,device=DEVICE)
                    LM_sentences.extend(LM_sentences_batch)
                if delete:
                    # Free GPU memory
                    object_detector_features=object_detector_features.to('cpu')
                    del object_detector_features
                    torch.cuda.empty_cache()
                return LM_sentences,selected_regions

       
    def filter_inputs_to_language_model(self, selection_classifier_targets, input_ids, attention_mask, object_detector_features,language_model_targets):
        '''
        Filters the inputs to the language model based on the outputs of the object detector and binary classifiers.

        Args:
            - selection_classifier_targets (Tensor):Binary Tensor of shape [batch_size x,29]
                Ground truth indicating whether a phrase exists in the region or not.
            - input_ids (Tensor): Input tensor for the language model.
            - attention_mask (Tensor): Attention mask for the language model.
            - object_detector_features (Tensor): Output features from the object detector.

        Returns:
            - valid_input_ids (Tensor): Input tensor for the language model.
            - valid_attention_mask (Tensor): Attention mask for the language model.
            - valid_region_features (Tensor): Output features from the object detector.
        '''
        # using gold standard labels to filter the input to the language model
        valid_input_ids = input_ids[selection_classifier_targets]
        valid_attention_mask = attention_mask[selection_classifier_targets]
        valid_region_features = object_detector_features[selection_classifier_targets]
        valid_labels=language_model_targets[selection_classifier_targets]
        return valid_input_ids, valid_attention_mask, valid_region_features,valid_labels
    
    # def post_processing(self,images,bboxes,bbox_labels):
    #     transform = CustomAugmentation(transform_type='test')
    #     for i in range(images.shape[0]):
    #         transformed = transform(image=images[i], bboxes=bboxes[i], class_labels=bbox_labels[i])
    #         transformed_image = transformed["image"]
    #         transformed_bboxes = transformed["bboxes"]
    #         transformed_bbox_labels = transformed["class_labels"]
    #         transformed_bboxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
    #         transformed_bbox_labels = torch.as_tensor(transformed_bbox_labels, dtype=torch.int64)
    #         object_detector_sample = {}
    #         object_detector_sample["image"]=transformed_image
    #         object_detector_sample["bboxes"] = transformed_bboxes
    #         object_detector_sample["bbox_labels"] = transformed_bbox_labels

        