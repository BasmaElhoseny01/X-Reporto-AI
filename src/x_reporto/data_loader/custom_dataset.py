import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch
from torchvision.transforms import functional as F
import pandas as pd 
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import ast
from torch.utils.data import Dataset, DataLoader
from src.x_reporto.data_loader.custom_augmentation import CustomAugmentation
from src.x_reporto.data_loader.tokenizer import Tokenizer
from src.language_model.GPT2.config import Config
import matplotlib.pylab as plb
class CustomDataset(Dataset):
    def __init__(self, dataset_path: str, transform_type:str ='train',checkpoint:str='healx/gpt-2-pubmed-medium'):
        self.dataset_path = dataset_path # path to csv file
        
        self.transform_type = transform_type
        self.transform = CustomAugmentation(transform_type=self.transform_type)
        self.checkpoint=checkpoint
        self.tokenizer = Tokenizer(self.checkpoint)
        self.data_info = pd.read_csv(dataset_path, header=None)       
        # remove the first row (column names)
        self.data_info = self.data_info.iloc[1:]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        try:
                
            # get the image path
            img_path = self.data_info.iloc[idx, 3]

            # read the image with parent path of current folder + image path
            # img_path = os.path.join("datasets/", img_path)
            img_path = os.path.join(os.getcwd(), img_path)
            # Fix Problem of \
            img_path = img_path.replace("\\", "/")
            img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)

            assert img is not None, f"Image at {img_path} is None"
 
            # get the bounding boxes
            bboxes = self.data_info.iloc[idx, 4]

            # convert the string representation of bounding boxes into list of list
            bboxes = eval(bboxes)

            # get the bbox_labels
            bbox_labels = self.data_info.iloc[idx, 5]

            # convert the string representation of labels into list
            bbox_labels = np.array(eval(bbox_labels))

            # get the bbox_labels
            bbox_phrases = self.data_info.iloc[idx, 6]
        
            try:
                # convert the string representation of labels into list
                bbox_phrases = eval(bbox_phrases)
            except Exception as e:
                # create a list of empty strings of size 29
                bbox_phrases = [""] * 29
            # get the bbox_labels
            bbox_phrase_exists = self.data_info.iloc[idx, 7]
            # get the bbox_labels
            bbox_is_abnormal = self.data_info.iloc[idx, 8]
            # tranform image
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=bbox_labels)
            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            transformed_bbox_labels = transformed["class_labels"]
            # convert the bounding boxes to tensor
            transformed_bboxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
            transformed_bbox_labels = torch.as_tensor(transformed_bbox_labels, dtype=torch.int64)
            #object_detector_targets
            object_detector_sample = {}
            object_detector_sample["image"]=transformed_image
            object_detector_sample["bboxes"] = transformed_bboxes
            object_detector_sample["bbox_labels"] = transformed_bbox_labels

            #classifier_targets
            # Safely evaluate the string and convert it to a Python list
            bbox_phrase_exists = ast.literal_eval(bbox_phrase_exists)

            # Convert the Python list to a PyTorch tensor
            bbox_phrase_exists = torch.tensor(bbox_phrase_exists, dtype=torch.bool)

            selection_classifier_sample= {}
            selection_classifier_sample["bbox_phrase_exists"]=bbox_phrase_exists

            #classifier_targets
            # Safely evaluate the string and convert it to a Python list
            bbox_is_abnormal = ast.literal_eval(bbox_is_abnormal)

            # Convert the Python list to a PyTorch tensor
            bbox_is_abnormal = torch.tensor(bbox_is_abnormal, dtype=torch.bool)
            
            abnormal_classifier_sample= {}
            abnormal_classifier_sample["bbox_is_abnormal"]=bbox_is_abnormal

            #language_model_targets
            language_model_sample={}
            try:
                tokenize_phrase = self.tokenizer(bbox_phrases)
            except Exception as e:
                tokenize_phrase = self.tokenizer([""] * 29)
            # get max sequence length in tokenized phrase
            max_seq_len = max([len(tokenize_phrase_lst) for tokenize_phrase_lst in tokenize_phrase["input_ids"]])
            language_model_sample["bbox_phrase"]=bbox_phrases
            padded_lists_by_pad_token = [tokenize_phrase_lst + [Config.pad_token_id] * (max_seq_len - len(tokenize_phrase_lst)) for tokenize_phrase_lst in tokenize_phrase["input_ids"]]
            padded_lists_by_ignore_token = [tokenize_phrase_lst + [Config.ignore_index] * (max_seq_len - len(tokenize_phrase_lst)) for tokenize_phrase_lst in tokenize_phrase["input_ids"]]
            language_model_sample["input_ids"]=padded_lists_by_pad_token
            language_model_sample["label_ids"]=padded_lists_by_ignore_token
            padded_mask = [mask_phrase_lst + [0] * (max_seq_len - len(mask_phrase_lst)) for mask_phrase_lst in tokenize_phrase["attention_mask"]]
            language_model_sample["attention_mask"]=padded_mask
            # convert the label to tensor
            language_model_sample["input_ids"] = torch.tensor(language_model_sample["input_ids"], dtype=torch.long)
            language_model_sample["label_ids"] = torch.tensor(language_model_sample["label_ids"], dtype=torch.long)
            language_model_sample["attention_mask"] = torch.tensor(language_model_sample["attention_mask"], dtype=torch.long)

            # print("end Tokenize")
            return object_detector_sample,selection_classifier_sample,abnormal_classifier_sample,language_model_sample
        except Exception as e:
            print(e)
            return None

    def collate_fn(self,batch):
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

            # check if batch_size >1
        if len(batch)>1:
            # pad phrases to the same length 
            # input_ids is list of tensors
            max_seq_len = max([len(tokenize_phrase_lst[0]) for tokenize_phrase_lst in input_ids])
            # each tensor in list input_ids is padded to max_seq_len
            new_input_ids=[]
            new_attention_mask=[]
            new_LM_targets=[]
            for i in range(len(input_ids)):
                # each tensor in list input_ids is padded to max_seq_len
                new_input_ids.append([])
                new_attention_mask.append([])
                new_LM_targets.append([])
                for j in range(len(input_ids[i])):
                    # concatenate the tensor with pad_token_id to the max_seq_len
                    new_input_ids[i].append(torch.cat((input_ids[i][j], torch.tensor([Config.pad_token_id] * (max_seq_len - len(input_ids[i][j])), dtype=torch.long))))
                    # concatenate the tensor with ignore_index to the max_seq_len
                    new_attention_mask[i].append(torch.cat((attention_mask[i][j], torch.tensor([0] * (max_seq_len - len(attention_mask[i][j])), dtype=torch.long))))
                    # concatenate the tensor with ignore_index to the max_seq_len
                    new_LM_targets[i].append(torch.cat((LM_targets[i][j], torch.tensor([Config.ignore_index] * (max_seq_len - len(LM_targets[i][j])), dtype=torch.long))))

            input_ids=new_input_ids
            attention_mask=new_attention_mask
            LM_targets=new_LM_targets
            # convert the list of tensors to tensor
            input_ids = [torch.stack(input_id) for input_id in input_ids]
            attention_mask = [torch.stack(mask) for mask in attention_mask]
            LM_targets = [torch.stack(target) for target in LM_targets]


        selection_classifier_targets=torch.stack(selection_classifier_targets)
        abnormal_classifier_targets=torch.stack(abnormal_classifier_targets)
        LM_targets=torch.stack(LM_targets)
        LM_inputs['input_ids']=torch.stack(input_ids)
        LM_inputs['attention_mask']=torch.stack(attention_mask)
        return images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets

if __name__ == "__main__":
    # create a dataset object
    dataset = CustomDataset("datasets/train.csv",transform_type='val')
    # create a dataloader object
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # get the first batch
    for object_detector_sample,selection_classifier_sample,abnormal_classifier_sample,language_model_sample in dataloader:
        print(object_detector_sample["image"].shape )
        break
    # display the first image
    print(object_detector_sample["image"].max(), object_detector_sample["image"].min())
    plb.imshow(object_detector_sample["image"][0][0],cmap="gray")
    plb.show()