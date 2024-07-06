import argparse
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from typing import List, Tuple, Dict, Any, Union, Optional
import evaluate

# Utility functions
from src.utils import plot_single_image
from config import OPERATION_MODE,OperationMode

# Models
from src.x_reporto.models.x_reporto_v1 import XReportoV1
from transformers import GPT2Tokenizer

from config import *
from src.denoiser.config import*
import numpy as np
# import asyncio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False


transform =  A.Compose(
                        [
                            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
                            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE,border_mode= cv2.BORDER_CONSTANT,value=0),

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
            # image=transform(image=image)["image"]
            # image= np.copy(image)

            image=transform(image=image)["image"]

            if image.dtype != np.float32:
                image = image.astype(np.float32)

            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
            image /= 255.0
            return image

class XReporto:
    x_reporto = None
    tokenizer = None
    bertScore = None
    # _lock = asyncio.Lock()
    def __init__(self):

        # check if the model is already loaded
        if XReporto.x_reporto is not None:
            print("Model Already Loaded")
            return
        print("Loading Model for first time")
        # Read the model
        XReporto.x_reporto = XReportoV1(object_detector_path="models/object_detector_best.pth",
                                abnormal_classifier_path="models/abnormal_classifier_best.pth",
                                region_classifier_path="models/region_classifier_best.pth",
                                language_model_path="models/LM_best.pth")
        
        XReporto.x_reporto.to(DEVICE)
        XReporto.tokenizer = GPT2Tokenizer.from_pretrained("healx/gpt-2-pubmed-medium")
        XReporto.bertScore = evaluate.load("bertscore")

        print("Device: ", DEVICE)
        print("Model Loaded Successfully")

    def object_detection(self,image_path):

        image = CustomDataset(img_path=image_path)        
        # Move the image to GPU
        # image = image.to(DEVICE) 
        image = torch.tensor(image).to(DEVICE)

        # Inference Pass
        XReporto.x_reporto.eval()
        with torch.no_grad():
            # Inference Pass
            bounding_boxes, deteced_classes =  XReporto.x_reporto(images=image, selected_models = [True,True,False,False])  

            # move the bounding boxes to cpu
            bounding_boxes = bounding_boxes.to('cpu')
        
        return bounding_boxes, deteced_classes

    def denoise_image(self,image_path):
        """
        Denoise the image

        Args:
        image_path (str): Path to the image file

        Returns:
        image: after denoising
        """
        image = CustomDataset(img_path=image_path)        
        # Move the image to GPU
        # image = image.to(DEVICE) 
        image = torch.tensor(image).to(DEVICE)

        # Inference Pass
        XReporto.x_reporto.eval()
        with torch.no_grad():
            # Inference Pass
            denoised_image =  XReporto.x_reporto(images=image, selected_models = [True,False,False,False])

            # squeeze the image from (1,1,w,h) to (w,h)
            denoised_image = np.squeeze(denoised_image)

        return denoised_image
        
    def generate_image_report(self,image_path):
        """
        Generate the report for the given image

        Args:
        image_path (str): Path to the image file

        Returns:
        image: after resizing and padding
        bounding_boxes: bounding boxes of the selected region
        abnormal_region: selected region
        lm_sentences_decoded: list of generated sentences
        report text to be saved later
        """
        # Read the image
        # image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)

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
        # image = image.to(DEVICE) 
        image = torch.tensor(image).to(DEVICE)

        # Inference Pass
        XReporto.x_reporto.eval()
        with torch.no_grad():
            # Inference Pass
            denoised_image, bounding_boxes, selected_region, abnormal_region, lm_sentences_encoded, detected_classes =  XReporto.x_reporto(images=image,use_beam_search=True)  

            # Decode the sentences
            lm_sentences_decoded=XReporto.tokenizer.batch_decode(lm_sentences_encoded,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            
            # Results
            image=image[0].to('cpu')
            bounding_boxes=bounding_boxes.to('cpu').numpy()
        
            # detected_classes = self.convert_boolean_classes_to_list(detected_classes)
            # # Bounding Boxes
            # plot_single_image(img = image.permute(1,2,0),boxes=bounding_boxes,grayscale=True,save_path='region.jpg')

        # generate the report text
        generated_sentences, report_text = self.fix_sentences(generated_sentences=lm_sentences_decoded)

        return bounding_boxes,detected_classes, generated_sentences, report_text


    def fix_sentences(self,generated_sentences: List[str]):
        """
        Fix the generated sentences
        1. Remove the duplicate sentences
        2. Remove the sentences with less than 3 words
        3. Remove the sentences with less than 3 characters
        4. Remove garbage sentences with random characters that has no meaning

        remove the sentences that has almost same meaning

        Args:
        generated_sentences (List[str]): List of generated sentences

        Returns:
        List[str]: List of fixed sentences
        str: Report text
        """
        
        # Remove the sentences with less than 3 words
        generated_sentences = [sentence for sentence in generated_sentences if len(sentence.split()) >= 3]

        # Remove the sentences with less than 3 characters
        generated_sentences = [sentence for sentence in generated_sentences if len(sentence) >= 3]

        # Remove the duplicate sentences that has almost same meaning
        # use bertScore function in evaluate
        # group the similar sentences
        similar_sentences = []
        final_sentences = []
        for sentence in generated_sentences:
            # check if the sentence is similar to any of the sentence in the similar_sentences
            is_similar = False
            i = 0
            for group in similar_sentences:
                for similar_sentence in group:
                    results = XReporto.bertScore.compute(
                            predictions=[sentence], 
                            references= [similar_sentence], 
                            lang="en", 
                            model_type="roberta-large"
                    )
                    if results["f1"][0] > 0.9:
                        is_similar = True
                        break

                if is_similar:
                    break
                i += 1
            if not is_similar:
                similar_sentences.append([sentence])
                final_sentences.append(sentence)
            else:
                # replace final sentence with current sentence if current sentence is longer
                if len(sentence) > len(final_sentences[i]):
                    final_sentences[i] = sentence
                
                # add the sentence to group
                similar_sentences[i].append(sentence)
        
        if DEBUG:
            # print each group and final sentence
            for i, group in enumerate(similar_sentences):
                print("Group: ", i)
                for sentence in group:
                    print(sentence)
                print("Final Sentence: ", final_sentences[i])
        # Remove garbage sentences with random characters that has no meaning
        
    
        # generate the report text
        report_text = ""
        for sentence in final_sentences:
            report_text += sentence + '\n'

        if DEBUG:
            print("Report Text: ", report_text)

        return final_sentences, report_text

    def convert_boolean_classes_to_list(self,classes):
        """
        Convert the boolean classes to list of classes

        Args:
        classes (torch.Tensor): Boolean tensor of classes

        Returns:
        List[int]: List of classes
        """

        # squeeze the classes
        classes = torch.squeeze(classes)
        detected_classes = []
        for i in range(classes.shape[0]):
            if classes[i]:
                detected_classes.append(i)

        return detected_classes


if __name__=="__main__":
    print("Basma")
    print("DEVICE",DEVICE)
    # Take image path from command line
    if OperationMode.INFERENCE.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Inference Mode")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()

    image_path = args.image_path
    print("inferencing input at",image_path)

    
    # Initialize the Inference class
    inference = XReporto()

    # Generate the report
    inference.generate_image_report(image_path=image_path)

    
    
    
# python -m src.inference.main "./datasets/images/00000001_000.png"
# python -m src.inference.main "datasets\mimic-cxr-jpg\files\p11\p11001469\s54076811\d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"