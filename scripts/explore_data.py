# Logging
from logger_setup import setup_logging
import logging

from datetime import datetime

import os
import argparse
import sys
import cv2
import pandas as pd 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms.functional import to_tensor




from torch.utils.data import  DataLoader
from src.x_reporto.data_loader.custom_dataset import CustomDataset
from config import PERIODIC_LOGGING

from torch.utils.tensorboard import SummaryWriter

# Set the random seed for reproducibility
torch.manual_seed(42)



    
# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class EXploreData():
    def __init__(self,tensor_board_writer:SummaryWriter,csv_path: str):
        self.tensor_board_writer=tensor_board_writer

        self.csv_path=csv_path

        self.read_csv_and_samples()
        # print("self.random_data",self.random_data)
        # print("self.random_indices",self.random_indices)


        # # create dataset
        # self.dataset= CustomDataset(csv_path,transform_type='val')
        # logging.info(f"Dataset loaded at {csv_path} Size: {len(self.dataset)} ")


       

        # # Data Loader
        # self.dataloader = DataLoader(dataset=self.dataset, collate_fn=collate_fn,batch_size=4, shuffle=False, num_workers=8)

        # logging.info(f"DataLoader Loaded Size: {len(self.dataloader)}")
   
    def read_csv_and_samples(self):
        # Reading CSV
        data_info = pd.read_csv(self.csv_path, header=None)

        # remove the first row (column names)
        data_info = data_info.iloc[1:]


        # Now, let's take random samples from the dataset
        num_samples = 5
        self.random_indices = torch.randperm(len(data_info))[:num_samples]
      
        self.random_data=data_info.iloc[self.random_indices-1]



    def show_csv_sample_images(self):          
        i=0
        for index in self.random_indices:
            index=index.item()

            img_path = self.random_data.loc[index, 3]

            # get the image path
            img_path = os.path.join(os.getcwd(), img_path)
            # Fix Problem of \
            img_path = img_path.replace("\\", "/")

            # read the image with parent path of current folder + image path
            img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            assert img is not None, f"Image at {img_path} is None"
         
            img = to_tensor(img)

            # [Tensor Board]: 
            self.tensor_board_writer.add_image(f'{index}', img)



            # # get the bounding boxes
            # bboxes = self.random_data.loc[index, 4]
            # # convert the string representation of bounding boxes into list of list
            # bboxes = eval(bboxes)



        





 
        
    def show_data_loader_sample_images(self):
        pass
       #     # images,labels=next(dataiter)  # Get First Batch
        #     for batch_idx,(images,object_detector_targets,selection_classifier_targets,abnormal_classifier_targets,LM_inputs,LM_targets) in tqdm(enumerate(self.dataloader)):
        #         print("batch_idx",batch_idx)
        #         # create grid of images
        #         img_grid = torchvision.utils.make_grid(images,nrow=4)

        #         # write to tensorboard
        #  ``       self.tensor_board_writer.add_image('', img_grid)

    def show_dataset_sample_images(self):
        pass
        # for indx,





       
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
    logging.info("Data Exploration")

    parser = argparse.ArgumentParser(description="Exploring Data.")
    parser.add_argument("--csv", help="Name of the csv file to sample from",default='./datasets/train.csv')

    args = parser.parse_args()

    csv_path=args.csv


    # Logging Configurations
    # log_config()
    logging.info(f"PERIODIC_LOGGING: {PERIODIC_LOGGING}")
    logging.info(f"csv_path: {csv_path}")

    # Creating tensorboard folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensor_board_folder_path=f"./tensor_boards/data_{current_datetime}"
    if not os.path.exists(tensor_board_folder_path):
        os.makedirs(tensor_board_folder_path)
        logging.info(f"Folder '{tensor_board_folder_path}' created successfully.")
    else:
        logging.info(f"Folder '{tensor_board_folder_path}' already exists.")
    
    # Tensor Board
    tensor_board_writer=SummaryWriter(tensor_board_folder_path)

    # Create an XReportoTrainer instance with the X-Reporto model
    explorer = EXploreData(tensor_board_writer=tensor_board_writer,csv_path=csv_path)

    # Show Images read from csv
    explorer.show_csv_sample_images()
    


    # create grid of images
    # img_grid = torchvision.utils.make_grid(images,nrow=4)

    # show images
    # matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    # writer.add_image('four_fashion_mnist_images', img_grid)


    # print(dataset[0])
    # print(dataset[0][0])
    # print(dataset[0][1])

    # object_detector_sample,selection_classifier_sample,abnormal_classifier_sample,language_model_sample=dataset[0]
    # print(object_detector_sample)






if __name__ == '__main__':
    # Call the setup_logging function at the beginning of your script
    setup_logging(log_file_path='./logs/explore_data.log',bash=True,periodic_logger=PERIODIC_LOGGING)

    try:
        # The main script runs here
        main()
    except Exception as e:
        # Log any exceptions that occur
        logging.exception("An error occurred",exc_info=True)
    

# python -m src.x_reporto.testing.X_reporto_testing.py