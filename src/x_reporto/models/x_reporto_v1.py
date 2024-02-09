import sys
import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional, List, Dict

from config import ModelStage,MODEL_STAGE,DEVICE,CONTINUE_TRAIN,TRAIN_RPN,LM_Batch_Size

# Utils 
from src.utils import load_model

# Modules
from src.object_detector.models.object_detector_factory import ObjectDetector

from src.binary_classifier.models.binary_classifier_selection_region_factory import BinaryClassifierSelectionRegion
from src.binary_classifier.models.binary_classifier_region_abnormal_factory import BinaryClassifierRegionAbnormal
from src.language_model.GPT2.gpt2_model import CustomGPT2
from src.language_model.GPT2.config import Config
class XReportoV1(nn.Module):
    """
    A modular model for object detection and binary classification.

    Attributes:
        - num_classes (int): The number of classes for object detection.

    Modules:
        - object_detector (ObjectDetectorWrapper): Object detector module.
        - binary_classifier_selection_region (BinaryClassifierSelectionRegionWrapper): Binary classifier for region selection.
        - binary_classifier_region_abnormal (BinaryClassifierRegionAbnormalWrapper): Binary classifier for abnormal region detection.
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
            config.d_model = 768
            config.d_ff1 = 768
            config.d_ff2 = 768
            config.d_ff3 = 768
            config.num_layers = 12
            config.vocab_size = 50257
            config.max_seq_len = 1024
            config.pretrained_model = "gpt2"
            image_config = Config()
            image_config.d_model = 1024
            image_config.d_ff1 = 1024
            image_config.d_ff2 = 1024
            image_config.d_ff3 = 768
            image_config.num_heads = 8
            image_config.num_layers = 6
            image_config.vocab_size = 50257
            image_config.max_seq_len = 1024
            image_config.dropout = 0.1
            self.language_model = CustomGPT2(config,image_config)
            # convert the model to half precision
            self.language_model.half()
            self.language_model.convert_to_half()

        if CONTINUE_TRAIN:
            if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value and TRAIN_RPN:
                    print("Loading object_detector [Trained RPN]....")
                    load_model(model=self.object_detector,name='object_detector_rpn')
            else:
                # Load full object detector
                print("Loading object_detector .....")
                load_model(model=self.object_detector,name='object_detector')


            if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                # Load the Region Selection Classifier to continue training
                print("Loading region_classifier .....")
                load_model(model=self.binary_classifier_selection_region,name='region_classifier')

                # Load the Abnormal Classifier to continue training
                print("Loading abnormal_classifier .....")
                load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier')
            
            if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                # Load Language Model to continue training
                pass
            
        else:
            if MODEL_STAGE==ModelStage.OBJECT_DETECTOR.value:
                if TRAIN_RPN:
                    pass
                else:
                    print("Loading object_detector [Trained RPN]....")
                    load_model(model=self.object_detector,name='object_detector_rpn')

            if MODEL_STAGE==ModelStage.CLASSIFIER.value or MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                # Load the object_detector to continue training
                print("Loading object_detector .....")
                load_model(model=self.object_detector,name='object_detector')


            if MODEL_STAGE==ModelStage.LANGUAGE_MODEL.value :
                # Load the Region Selection Classifier to start training
                print("Loading region_classifier .....")
                load_model(model=self.binary_classifier_selection_region,name='region_classifier')

                # Load the Abnormal Classifier to start training
                print("Loading abnormal_classifier .....")
                load_model(model=self.binary_classifier_region_abnormal,name='abnormal_classifier')
      


            

    def forward(self,images: Tensor ,input_ids=None,attention_mask=None, object_detector_targets: Optional[List[Dict[str, Tensor]]] = None, selection_classifier_targets: Tensor=None,abnormal_classifier_targets: Tensor = None,language_model_targets: Tensor= None,batch=None,index=None,delete=False):
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
       '''
        stop=False
        if self.training:
            # Training
            # Stage(1) Object Detector
            print("Before object detector")
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
                return object_detector_losses,0
            # Stage(2) Binary Classifier
            print("Before binary classifier selection region")
            object_detector_detected_classes=object_detector_detected_classes.to(DEVICE)
            selection_classifier_losses,_,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,_=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            
            if delete:
                # free gpu memory
                abnormal_classifier_targets=abnormal_classifier_targets.to('cpu')
                del abnormal_classifier_targets
                torch.cuda.empty_cache()
           
            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,0
            
            # valid_input_ids, valid_attention_mask, valid_region_features=self.get_valid_decoder_input_for_training(object_detector_detected_classes, selection_classifier_targets, input_ids, attention_mask, object_detector_features)
            input_ids, attention_mask, object_detector_features = self.filter_inputs_to_language_model(selection_classifier_targets, input_ids, attention_mask, object_detector_features)
            
            if delete:
                # free gpu memory
                selection_classifier_targets=selection_classifier_targets.to('cpu')
                object_detector_detected_classes=object_detector_detected_classes.to('cpu')
                del selection_classifier_targets
                del object_detector_detected_classes
                torch.cuda.empty_cache()
            
            print("Before language model")

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

            return object_detector_losses,selection_classifier_losses,abnormal_binary_classifier_losses,LM_output[0]
        
        else: # Validation (or inference) mode
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
                return object_detector_losses,object_detector_boxes,object_detector_detected_classes
            
            # Stage(2) Binary Classifier
            selection_classifier_losses,selected_regions,_=self.binary_classifier_selection_region(object_detector_features,object_detector_detected_classes,selection_classifier_targets)
            abnormal_binary_classifier_losses,predicted_abnormal_regions=self.binary_classifier_region_abnormal(object_detector_features,object_detector_detected_classes,abnormal_classifier_targets)
            if delete:
                # free gpu memory
                selection_classifier_targets=selection_classifier_targets.to('cpu')
                del selection_classifier_targets
                torch.cuda.empty_cache()

            if MODEL_STAGE == ModelStage.CLASSIFIER.value:
                return object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions
           
            # Stage(3) Language Model
                      
            print("Before language model")       
            input_ids, attention_mask, object_detector_features = self.filter_inputs_to_language_model(selected_regions, input_ids, attention_mask, object_detector_features)
            print("here is the problem ",len(input_ids))
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

            return object_detector_losses,object_detector_boxes,object_detector_detected_classes,selection_classifier_losses,selected_regions,abnormal_binary_classifier_losses,predicted_abnormal_regions,LM_output[0],stop
    
    def filter_inputs_to_language_model(self, selection_classifier_targets, input_ids, attention_mask, object_detector_features):
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
        print("valid_input_ids",valid_input_ids.size())
        print("valid_attention_mask",valid_attention_mask.size())
        print("valid_region_features",valid_region_features.size())
        return valid_input_ids, valid_attention_mask, valid_region_features