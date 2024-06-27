import argparse
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Utility functions
from config import OPERATION_MODE,OperationMode

# Models
from src.heat_map_U_ones.models.heat_map import HeatMap


from config import *
from src.denoiser.config import*
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# def CustomDataset(img_path: str): 
#             img_path = os.path.join(os.getcwd(), img_path)
#             # replace \ with / for windows
#             img_path = img_path.replace("\\", "/")
#             image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
#             image=np.array(image).astype("float32")
#             if image is  None:
#                 assert image is not None, f"Image at {img_path} is None"
#             # image=transform(image=image)["image"]
#             # image= np.copy(image)

#             image=transform(image=image)["image"]

#             if image.dtype != np.float32:
#                 image = image.astype(np.float32)

#             image = np.expand_dims(image, axis=0)
#             image = np.expand_dims(image, axis=0)
#             image /= 255.0
#             return image
           
class HeatMapInference:
    def __init__(self):

        # Create the model
        self.heat_map_model=HeatMap()

        # Load the model
        self.heat_map_model.load_state_dict(torch.load('models/heat_map_4/heat_map_best.pth'))

        # Move to Device
        self.heat_map_model.to(DEVICE)
        
        # Evaluation Mode
        self.heat_map_model.eval()

        print("Model Loaded")
        
        self.heat_map_transform=A.Compose([
            A.LongestMaxSize(max_size=HEAT_MAP_IMAGE_SIZE, interpolation=cv2.INTER_AREA),

            A.PadIfNeeded(min_height=HEAT_MAP_IMAGE_SIZE, min_width=HEAT_MAP_IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
            ToTensorV2(p=1.0)
        ])

    
    def _load_and_process_image(self, img_path: str):

        # Getting image path  with parent path of current folder + image path
        img_path = os.path.join(os.getcwd(), img_path)

        # Fix Image Path
        img_path = img_path.replace("\\", "/")

        # Read Image
        image = cv2.imread(img_path,cv2.IMREAD_COLOR)
        assert image is not None, f"Image at {img_path} is None"

        # convert the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #(3056, 2544, 3) [0-255]

        # to float32 
        image=np.array(image).astype("float32")

        # Apply Transformations
        image=self.heat_map_transform(image=image)["image"]

        return image

   
    def generate_template_based_report(self,image_path,bool):
        '''
        Generate Heat Map for the given image path + Template Based Report

        i/p: image Path
        o/p : Heat Map Image , 8 classification (+ the confidence), Severity, Template Based Report
        Options:- Weighted Summations of ones
                - Get Rank of findings by a Doctor
                - Summation of probabilities
                - Summation of Areas of the heat map locations 
        '''
        print("Heat Map Generation")

        # Load the image
        image=self._load_and_process_image(img_path=image_path)
        print(image)

        with torch.no_grad():
            image=image.to(DEVICE)

            # Forward Pass
            _,y_scores,_=self.heat_map_model(image)


            # Results
            image = image.to("cpu")
            confidence = y_scores.to("cpu")
            

if __name__=="__main__":
    # Check Operation Mode
    if OperationMode.INFERENCE.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Inference Mode")
    
    # Take image path from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()
    image_path = args.image_path
    
    # Initialize the Inference class
    inference = HeatMapInference()

    # Generate Template Based Report & Heat Map
    inference.generate_template_based_report()

          
    
    
    
# python -m src.inference.main "./datasets/images/00000001_000.png"
# python -m src.inference.main "datasets\mimic-cxr-jpg\files\p11\p11001469\s54076811\d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"