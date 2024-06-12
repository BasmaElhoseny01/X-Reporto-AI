import argparse
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Utility functions
from src.utils import plot_single_image
from config import OPERATION_MODE,OperationMode

# Models
from src.x_reporto.models.x_reporto_v1 import XReportoV1
from transformers import GPT2Tokenizer

from config import *
from src.denoiser.config import*
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform =  A.Compose(
                        [
                            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                        ]
                    )
transform2 =  A.Compose(
                      [
                          A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE,border_mode= cv2.BORDER_CONSTANT,value=0),
                      ]
                  )

transform3 = A.Compose([
                        A.Normalize(mean=0.474, std=0.301),
                        ToTensorV2(),
                      ]
                 )

def CustomDataset(img_path: str): 
            img_path = os.path.join(os.getcwd(), img_path)
            # replace \ with / for windows
            img_path = img_path.replace("\\", "/")
            image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            image=np.array(image).astype("float32")
            if image is  None:
                assert image is not None, f"Image at {img_path} is None"
            image=transform(image=image)["image"]
            image= np.copy(image)

            image=transform2(image=image)["image"]

            if image.dtype != np.float32:
                image = image.astype(np.float32)

            image = np.expand_dims(image, axis=0)
            image /= 255.0
            return image

class Inference:
    def __init__(self):

        # Read the model
        self.x_reporto = XReportoV1(object_detector_path="models/object_detector.pth",
                               region_classifier_path="models/binary_classifier_selection_region.pth",
                               language_model_path="models/LM.pth")
        
        self.x_reporto.to(DEVICE)
        print("Model Loaded")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("healx/gpt-2-pubmed-medium")


                               

    def generate_image_report(self,image_path):
        # Read the image
        image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        #print(image.shape)

        # transform = A.Compose([
        #     A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
        #     A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
        #     A.Normalize(mean=0.474, std=0.301),
        #     ToTensorV2(),
        # ])
        # image = transform(image=image)['image']


        # # Add batch dimension
        # image = image.unsqueeze(0)

        image = CustomDataset(img_path=image_path)
        
        # Move the image to GPU
        image = image.to(DEVICE)   

        # Inference Pass
        self.x_reporto.eval()
        with torch.no_grad():
            bounding_boxes,lm_sentences_encoded=self.x_reporto(images=image,use_beam_search=False)            
            lm_sentences_decoded=self.tokenizer.batch_decode(lm_sentences_encoded,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            
           
            
            # Results
            image=image[0].to('cpu')
            bounding_boxes=bounding_boxes.to('cpu')
            
            # Bounding Boxes
            plot_single_image(img=image.permute(1,2,0),boxes=bounding_boxes,grayscale=True,save_path='region.jpg')

            # Report
            report_path='report.txt'
            with open(report_path, "w") as file:
                # Iterate over each sentence in the list
                for sentence in lm_sentences_decoded:
                    file.write(sentence + "\n")
                print("Report Saved Successfully at: ",report_path)

        
        # Input is Image
        # Output is Image with bounding box
        # Selected Regions / Abnormal Region
        # Report

    #     pass

if __name__=="__main__":
    # Take image path from command line
    if OperationMode.INFERENCE.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Inference Mode")
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()

    image_path = args.image_path
    print("inferencing input at",image_path)

    
    # Initialize the Inference class
    inference = Inference()

    # Generate the report
    inference.generate_image_report(image_path=image_path)

          
    
    
    
# python -m src.inference.main "./datasets/images/00000001_000.png"
# python -m src.inference.main "datasets\mimic-cxr-jpg\files\p11\p11001469\s54076811\d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"