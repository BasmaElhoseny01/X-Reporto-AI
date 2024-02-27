import sys
import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional, List, Dict

from config import *
# Utils 
from src.utils import load_model

# Modules
from src.object_detector.models.object_detector_factory import ObjectDetector
from src.binary_classifier.models.binary_classifier_selection_region_factory import BinaryClassifierSelectionRegion
from src.binary_classifier.models.binary_classifier_region_abnormal_factory import BinaryClassifierRegionAbnormal
from src.language_model.GPT2.gpt2_model import CustomGPT2
from src.language_model.GPT2.config import Config
from transformers import GPT2Tokenizer
import logging
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

    def __init__(self):
        super().__init__()
        self.num_classes=30

        self.object_detector = ObjectDetector().create_model()

        if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
            self.binary_classifier_selection_region = BinaryClassifierSelectionRegion().create_model()
            self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal().create_model()
        if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value:
            config = Config()
            image_config = Config()
            self.language_model = CustomGPT2(config,image_config)
            # convert the model to half precision
            # self.language_model.half()
            # self.language_model.convert_to_half()

        if RECOVER==True:
            # Don't Load any module the check point will be loaded Later :
            logging.debug("No Modules are loaded in x_reporto_v1 due to Recovery Mode")

        elif OPERATION_MODE==OperationMode.TRAINING.value:
            if CONTINUE_TRAIN:
                if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value and TRAIN_RPN:
                        
                        logging.info("Loading object_detector [Trained RPN]....")
                        load_model(model=self.object_detector,name='object_detector_rpn_best')
                else:
                    # Load full object detector
                    logging.info("Loading object_detector .....")
                    load_model(model=self.object_detector,name='object_detector_best')
                    
                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load the Region Selection Classifier to continue training
                    logging.info("Loading region_classifier .....")
                    load_model(model=self.binary_classifier_selection_region,name='region_classifier_best')

                    # Load the Abnormal Classifier to continue training
                    logging.info("Loading abnormal_classifier .....")
                    load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier_best')

                    # Freezing Object Detector Model [including Backbone, RPN, RoI Heads]
                    for param in self.object_detector.object_detector.parameters():
                        param.requires_grad = False
                        
                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load Language Model to continue training
                    logging.info("Loading language_model .....")
                    load_model(model=self.language_model,name='LM_best')

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
                    else:
                        logging.info("Loading object_detector [Trained RPN]....")
                        load_model(model=self.object_detector,name='object_detector_rpn_best')
                    

                if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load the object_detector to continue training
                    logging.info("Loading object_detector .....")
                    load_model(model=self.object_detector,name='object_detector_best')
                    # Freezing Object Detector Model [including Backbone, RPN, RoI Heads]
                    for param in self.object_detector.object_detector.parameters():
                        param.requires_grad = False


                if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                    # Load the Region Selection Classifier to start training
                    logging.info("Loading region_classifier .....")
                    load_model(model=self.binary_classifier_selection_region,name='region_classifier_best')
                    # Freezing Selection Region Binary Classifier
                    for param in self.binary_classifier_selection_region.selection_binary_classifier.parameters():
                        param.requires_grad = False

                    # Load the Abnormal Classifier to start training
                    logging.info("Loading abnormal_classifier .....")
                    load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier_best')
                    # Freezing Abnormal Region Binary Classifier
                    for param in self.binary_classifier_region_abnormal.abnormal_binary_classifier.parameters():
                        param.requires_grad = False
                        # if  GENERATE_REPORT:
                        #     load_model(model=self.language_model,name='LM_best')

        elif OPERATION_MODE==OperationMode.VALIDATION.value or OPERATION_MODE==OperationMode.TESTING.value:
            if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                logging.info("Loading object_detector .....")
                load_model(model=self.object_detector,name='object_detector_best')

            elif MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                logging.info("Loading object_detector .....")
                load_model(model=self.object_detector,name='object_detector_best')
                logging.info("Loading region_classifier .....")
                load_model(model=self.binary_classifier_selection_region,name='region_classifier_best')
                logging.info("Loading abnormal_classifier .....")
                load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier_best')

            elif MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                logging.info("Loading language_model .....")
                load_model(model=self.language_model,name='LM_best')         


            

    def forward(self,images: Tensor ,input_ids=None,attention_mask=None, object_detector_targets: Optional[List[Dict[str, Tensor]]] = None, selection_classifier_targets: Tensor=None,abnormal_classifier_targets: Tensor = None,language_model_targets: Tensor= None,batch:Optional[int]=None,index:Optional[int]=None,delete:Optional[bool]= False,generate_sentence :Optional[bool]=False,use_beam_search :Optional[bool] = False,validate_during_training:Optional[bool]=False):
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
        if OPERATION_MODE==OperationMode.TRAINING.value and self.training:
            # Training
            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
            
            if delete:
                # Free GPU memory 
                images=images.to('cpu')
                # move object_detector_targets to cpu
                for i in range(len(object_detector_targets)):
                    object_detector_targets[i]['boxes']=object_detector_targets[i]['boxes'].to('cpu')
                    object_detector_targets[i]['labels']=object_detector_targets[i]['labels'].to('cpu')
                del images
                del object_detector_targets
                torch.cuda.empty_cache()

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,0,0,0
            # Stage(2) Binary Classifier
            object_detector_detected_classes=object_detector_detected_classes.to(DEVICE)
            selection_classifier_losses,_,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,_=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            
            if delete:
                # free gpu memory
                abnormal_classifier_targets=abnormal_classifier_targets.to('cpu')
                object_detector_detected_classes=object_detector_detected_classes.to('cpu')
                del abnormal_classifier_targets
                del object_detector_detected_classes
                torch.cuda.empty_cache()
           
            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,0
            # Stage(3) Language Model
            valid_input_ids, valid_attention_mask, valid_object_detector_features ,valid_labels= self.filter_inputs_to_language_model(selection_classifier_targets, input_ids, attention_mask, object_detector_features,language_model_targets)
            if delete or True:
                selection_classifier_targets=selection_classifier_targets.to('cpu')
                del selection_classifier_targets
                torch.cuda.empty_cache()
            if index>=len(valid_input_ids):
                return 0,0,0,0,0,0,0,0,True
            if (index+LM_Batch_Size) >= len(valid_input_ids):
                stop=True
            LM_output=self.language_model(input_ids=valid_input_ids[index:index+LM_Batch_Size,:],image_hidden_states=valid_object_detector_features[index:index+LM_Batch_Size,:],attention_mask=valid_attention_mask[index:index+LM_Batch_Size,:],labels=valid_labels[index:index+LM_Batch_Size,:])
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

            return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_output[0],stop
       
        if OPERATION_MODE==OperationMode.VALIDATION.value or validate_during_training:
            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
            if delete:
                # Free GPU memory 
                images=images.to('cpu')
                # move object_detector_targets to cpu
                for i in range(len(object_detector_targets)):
                    object_detector_targets[i]['boxes']=object_detector_targets[i]['boxes'].to('cpu')
                    object_detector_targets[i]['labels']=object_detector_targets[i]['labels'].to('cpu')
                del images
                del object_detector_targets
                torch.cuda.empty_cache()

            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,0,0,0
            
            # Stage(2) Binary Classifier
            selection_classifier_losses,selected_regions,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,predicted_abnormal_regions=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            if delete:
                # free gpu memory
                selection_classifier_targets=selection_classifier_targets.to('cpu')
                del selection_classifier_targets
                torch.cuda.empty_cache()

            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,0
           
            # Stage(3) Language Model
            input_ids, attention_mask, object_detector_features = self.filter_inputs_to_language_model(selected_regions, input_ids, attention_mask, object_detector_features)
            if index>=len(input_ids):
                return 0,0,0,0,0,0,0,0,True
            if (index+LM_Batch_Size) >= len(input_ids):
                stop=True
            LM_output=self.language_model(input_ids=input_ids[index:index+LM_Batch_Size,:],image_hidden_states=object_detector_features[index:index+LM_Batch_Size,:],attention_mask=attention_mask[index:index+LM_Batch_Size,:],labels=language_model_targets[batch][index:index+LM_Batch_Size,:])
            if delete:
                # Free GPU memory
                object_detector_features=object_detector_features.to('cpu')
                input_ids=input_ids.to('cpu')
                attention_mask=attention_mask.to('cpu')
                del object_detector_features
                del input_ids
                del attention_mask
                torch.cuda.empty_cache()
            return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_output[0],stop
        
        if OPERATION_MODE==OperationMode.EVALUATION.value:
            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
            if delete:
                # Free GPU memory 
                images=images.to('cpu')
                # move object_detector_targets to cpu
                for i in range(len(object_detector_targets)):
                    object_detector_targets[i]['boxes']=object_detector_targets[i]['boxes'].to('cpu')
                    object_detector_targets[i]['labels']=object_detector_targets[i]['labels'].to('cpu')
                del images
                del object_detector_targets
                torch.cuda.empty_cache()
            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,object_detector_boxes,object_detector_detected_classes,None,None,None,None,None,None,None
            
            # Stage(2) Binary Classifier
            selection_classifier_losses,selected_regions,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,predicted_abnormal_regions=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            if delete:
                # free gpu memory
                selection_classifier_targets=selection_classifier_targets.to('cpu')
                del selection_classifier_targets
                torch.cuda.empty_cache()

            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions,None,None,None
            
            # # Stage(3) Language Model
            # input_ids, attention_mask, object_detector_features = self.filter_inputs_to_language_model(selected_regions, input_ids, attention_mask, object_detector_features)
            # if index>=len(input_ids):
            #     return 0,0,0,0,0,0,0,0,True
            # if (index+LM_Batch_Size) >= len(input_ids):
            #     stop=True
            # LM_output=self.language_model(input_ids=input_ids[index:index+LM_Batch_Size,:],image_hidden_states=object_detector_features[index:index+LM_Batch_Size,:],attention_mask=attention_mask[index:index+LM_Batch_Size,:],labels=language_model_targets[batch][index:index+LM_Batch_Size,:])
            # if delete:
            #     # Free GPU memory
            #     object_detector_features=object_detector_features.to('cpu')
            #     input_ids=input_ids.to('cpu')

        if OPERATION_MODE==OperationMode.TESTING.value:
            # Stage(1) Object Detector
            object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
            if delete:
                # Free GPU memory 
                images=images.to('cpu')
                # move object_detector_targets to cpu
                for i in range(len(object_detector_targets)):
                    object_detector_targets[i]['boxes']=object_detector_targets[i]['boxes'].to('cpu')
                    object_detector_targets[i]['labels']=object_detector_targets[i]['labels'].to('cpu')
                del images
                del object_detector_targets
                torch.cuda.empty_cache()
            if MODEL_STAGE == ModelStage.OBJECT_DETECTOR.value:
                return object_detector_losses,object_detector_boxes,object_detector_detected_classes,None,None,None,None,None,None,None
            
            # Stage(2) Binary Classifier
            selection_classifier_losses,selected_regions,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,predicted_abnormal_regions=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            if delete:
                # free gpu memory
                selection_classifier_targets=selection_classifier_targets.to('cpu')
                del selection_classifier_targets
                torch.cuda.empty_cache()

            # if MODEL_STAGE == ModelStage.CLASSIFIER.value:
            #     return object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions
           
            # # Stage(3) Language Model
            # input_ids, attention_mask, object_detector_features = self.filter_inputs_to_language_model(selected_regions, input_ids, attention_mask, object_detector_features)
            # if index>=len(input_ids):
            #     return 0,0,0,0,0,0,0,0,True
            # if (index+LM_Batch_Size) >= len(input_ids):
            #     stop=True
            # LM_output=self.language_model(input_ids=input_ids[index:index+LM_Batch_Size,:],image_hidden_states=object_detector_features[index:index+LM_Batch_Size,:],attention_mask=attention_mask[index:index+LM_Batch_Size,:],labels=language_model_targets[batch][index:index+LM_Batch_Size,:])
            # if delete:
            #     # Free GPU memory
            #     object_detector_features=object_detector_features.to('cpu')
            #     input_ids=input_ids.to('cpu')
            #     attention_mask=attention_mask.to('cpu')
            #     del object_detector_features
            #     del input_ids
            #     del attention_mask
            #     torch.cuda.empty_cache()

            return object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions,LM_output[0],LM_output[1],stop
    
           
        # if generate_sentence:
                # object_detector_losses,object_detector_boxes,object_detector_detected_classes,object_detector_features = self.object_detector(images=images, targets=object_detector_targets)
                # if delete:
                #     # Free GPU memory 
                #     images=images.to('cpu')
                #     # move object_detector_targets to cpu
                #     # for i in range(len(object_detector_targets)):
                #     #     object_detector_targets[i]['boxes']=object_detector_targets[i]['boxes'].to('cpu')
                #     #     object_detector_targets[i]['labels']=object_detector_targets[i]['labels'].to('cpu')
                #     del images
                #     # del object_detector_targets
                #     torch.cuda.empty_cache()
                #     # Stage(2) Binary Classifier
                # selection_classifier_losses,selected_regions,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
                # if delete:
                #         # free gpu memory
                #         torch.cuda.empty_cache()
                # selected_regions=torch.ones_like(selected_regions)
                # object_detector_features = object_detector_features[selected_regions]
                # # if (index+LM_Batch_Size) >= object_detector_features.shape[0]-1:
                # #     stop=True
                # if use_beam_search:
                #     LM_sentencses=self.language_model.beam_search(max_length=50,image_hidden_states=object_detector_features[index:index+LM_Batch_Size,:],beam_size =6,device=DEVICE,debug=False)
                # else:
                #     LM_sentencses=self.language_model.generate(max_length=50,image_hidden_states=object_detector_features[index:index+LM_Batch_Size,:],greedy=True,device=DEVICE)
                
                # # LM_output=self.language_model(input_ids=input_ids[index:index+LM_Batch_Size,:],image_hidden_states=object_detector_features[index:index+LM_Batch_Size,:],attention_mask=attention_mask[index:index+LM_Batch_Size,:],labels=language_model_targets[batch][index:index+LM_Batch_Size,:])
                # if delete:
                #     # Free GPU memory
                #     object_detector_features=object_detector_features.to('cpu')
                #     del object_detector_features
                #     torch.cuda.empty_cache()

                # return LM_sentencses,stop

       
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