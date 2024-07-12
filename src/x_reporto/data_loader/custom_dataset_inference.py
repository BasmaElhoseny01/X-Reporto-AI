import numpy as np
import os
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.denoiser.data_loader.generate_noise import *
import matplotlib.pylab as plb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *
import cv2
from src.denoiser.config import*
from src.utils import plot_single_image
from src.language_model.GPT2.config import Config
import ast
from src.x_reporto.data_loader.tokenizer import Tokenizer

class CustomDataset(Dataset):
    def __init__(self, csv_file_path: str,checkpoint:str='healx/gpt-2-pubmed-medium'):
        self.data_info = pd.read_csv(csv_file_path, header=None)
        self.data_info =self.data_info.iloc[1:]
        self.checkpoint=checkpoint
        self.tokenizer = Tokenizer(self.checkpoint)
        self.transform =  A.Compose(
                        [
                            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                        ],bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
                    )
        self.transform2_box =  A.Compose(
                        [
                            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE,border_mode= cv2.BORDER_CONSTANT,value=0),
                        ],bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
                    )
        self.transform2 =  A.Compose(
                        [
                            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE,border_mode= cv2.BORDER_CONSTANT,value=0),
                        ]
                    )
        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        try:
            img_path =self.data_info.iloc[idx, 3]
            img_path = os.path.join(os.getcwd(), img_path)
            # replace \ with / for windows
            img_path = img_path.replace("\\", "/")
            image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            image=np.array(image).astype("float32")
            if image is  None:
                assert image is not None, f"Image at {img_path} is None"
            # get the bounding boxes
            bboxes = self.data_info.iloc[idx, 4]
            # convert the string representation of bounding boxes into list of list
            bboxes = eval(bboxes)
            # get the bbox_labels
            bbox_labels = self.data_info.iloc[idx, 5]
            # convert the string representation of labels into list
            bbox_labels = np.array(eval(bbox_labels))
            # tranform image
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=bbox_labels)
            image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            transformed_bbox_labels = transformed["class_labels"]

            choice= np.random.choice([0,1,2,3,4,5])
            # image=self.transform(image=image)["image"]
            if choice == 0:
                image,label= add_block_pixel_noise(image, probability=0.05)
            elif choice == 1:
                image,label= add_convolve_noise(image, sigma=1, sigma_noise=18) 
            elif choice == 2:
                image,label= add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
            elif choice == 3:
                image,label= add_gaussian_projection_noise(image, sigma=20)
            else:
                image,label= np.copy(image),np.copy(image)

            transform2_box = self.transform2_box(image=label, bboxes=transformed_bboxes, class_labels=transformed_bbox_labels)
            image=self.transform2(image=image)["image"]
            label=transform2_box["image"]
            transformed_bboxes = transform2_box["bboxes"]
            transformed_bbox_labels = transform2_box["class_labels"]
            # convert the bounding boxes to tensor
            transformed_bboxes = torch.as_tensor(transformed_bboxes, dtype=torch.float32)
            transformed_bbox_labels = torch.as_tensor(transformed_bbox_labels, dtype=torch.int64)
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if label.dtype != np.float32:
                label = label.astype(np.float32)
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            #Normalize the image and label, normlize boxes
            image /= 255.0
            label /= 255.0

            denoiser_sample = {}
            denoiser_sample['image']=image
            denoiser_sample['label']=label
            #object_detector_targets
            object_detector_sample = {}
            object_detector_sample["bboxes"] = transformed_bboxes
            object_detector_sample["bbox_labels"] = transformed_bbox_labels

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
            print("sentences",language_model_sample["label_ids"])
            return denoiser_sample, object_detector_sample,selection_classifier_sample,abnormal_classifier_sample,language_model_sample
        except Exception as e:
            print(e)
            return None
        
        
if __name__ == "__main__":
    # create a dataset object
    transform_test =  A.Compose(
                        [
                            A.Normalize(mean=MEAN, std=STD),
                            ToTensorV2(),
                        ]
                    )
    dataset = CustomDataset("datasets/train.csv")
    # create a dataloader object
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # get the first batch
    for denoiser_sample, object_detector_sample,_,_,_ in dataloader:
        print(denoiser_sample['image'].shape, denoiser_sample['label'].shape)
        label=denoiser_sample['label'][0]
        image=denoiser_sample['image'][0]
        bounding_boxes=object_detector_sample['bboxes'][0]
        label=transform_test(image=label)["image"]
        print(image.max(), image.min())
        bounding_boxes=bounding_boxes.to('cpu')
        # # Bounding Boxes
        image=image.to('cpu')
        plot_single_image(img=image.permute(1,2,0),boxes=bounding_boxes,grayscale=True,save_path='region.jpg')
        label=np.array(label)
        print(label.max(), label.min())

        plot_single_image(img=label.permute(1,2,0),boxes=bounding_boxes,grayscale=True,save_path='region_norm.jpg')
        break

# python -m src.x_reporto.data_loader.custom_dataset_inference