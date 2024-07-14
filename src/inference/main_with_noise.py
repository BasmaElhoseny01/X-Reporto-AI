import argparse
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

# Utility functions
from src.utils import plot_single_image
from config import OPERATION_MODE,OperationMode

# Models
from src.x_reporto.models.x_reporto_v1 import XReportoV1
from transformers import GPT2Tokenizer
from src.denoiser.data_loader.generate_noise import *



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Inference:
    def __init__(self):

        # Read the model
        self.x_reporto = XReportoV1(object_detector_path="models/object_detector.pth",
                               region_classifier_path="models/binary_classifier_selection_region.pth",
                               language_model_path="models/LM.pth")
        
        self.x_reporto.to(DEVICE)
        print("Model Loaded")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("healx/gpt-2-pubmed-medium")
        self.transform =  A.Compose(
                        [
                            A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
                        ]
                    )
        self.transform2 =  A.Compose(
                        [
                            A.PadIfNeeded(min_height=512, min_width=512,border_mode= cv2.BORDER_CONSTANT,value=0),
                        ]
                    )

    def generate_image_report(self,image_path):
       
        img_path = img_path.replace("\\", "/")
        image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        image=np.array(image).astype("float32")
        if image is  None:
            assert image is not None, f"Image at {img_path} is None"
            
        choice= np.random.choice([0,1,2,3,4,5])
        image=self.transform(image=image)["image"]
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

        image=self.transform2(image=image)["image"]
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