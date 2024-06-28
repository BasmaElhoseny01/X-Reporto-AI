import argparse
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Utility functions
from config import *

# Models
from src.heat_map_U_ones.models.heat_map import HeatMap



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
        self.heat_map_model.load_state_dict(torch.load('models/heat_map.pth'))

        # Move to Device
        self.heat_map_model.to(DEVICE)
        
        # Evaluation Mode
        self.heat_map_model.eval()

        # Optimal Thrsholds
        self.optimal_thresholds=self.heat_map_model.optimal_thresholds

        # Weights for Last Layer
        self.weights = list(self.heat_map_model.model.classifier.parameters())[-2] #torch.Size([8, 1024])

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #(3056, 2544, 3) [0-255]

        # to float32 
        image=np.array(image).astype("float32")

        # Apply Transformations
        image=self.heat_map_transform(image=image)["image"]
              
        # Add batch dimension
        image = image.unsqueeze(0)

        return image
 
    def infer(self,image_path,heatmap_type:str):
        '''
        Generate Heat Map for the given image path + Template Based Report
        heatmap_type: cam or grid_cam

        i/p: image Path
        o/p : Heat Map Image , 8 classification (+ the confidence), Severity, Template Based Report
        Options:- Weighted Summations of ones
                - Get Rank of findings by a Doctor
                - Summation of probabilities
                - Summation of Areas of the heat map locations 
        '''
        # Load the image
        image=self._load_and_process_image(img_path=image_path)

        with torch.no_grad():
            image=image.to(DEVICE)

            # Forward Pass
            _,y_scores,features=self.heat_map_model(image)

            # Results
            image = image.to("cpu").squeeze(0).data.numpy().transpose(1, 2, 0)
            confidence = y_scores.to("cpu").squeeze(0).tolist()
            features=features.to("cpu").squeeze(0)

            labels=[]
            for i,class_confidence in enumerate(confidence):
              label=1*(class_confidence>self.optimal_thresholds[i])
              labels.append(label)

            heatmaps=self.generate_heat_maps(features,heatmap_type)

            severity=self.compute_severity(labels,confidence)

            template_based_report=self.generate_template_based_report(labels=labels,confidence=confidence)

            return image,heatmaps,labels,confidence,severity,template_based_report

    def generate_template_based_report(self,labels,confidence):
      report=[]
      # TODO Add Info About the Grey Area
      template_positive = "The patient has {condition}."
      template_negative = "The patient does not have {condition}."

      for i, label in enumerate(labels):
          confidence_percent = confidence[i] * 100
          if label == 1:
              report.append(template_positive.format(condition=CLASSES[i], confidence=confidence_percent))
          else:
              report.append(template_negative.format(condition=CLASSES[i], confidence=confidence_percent))
      return report

    def compute_severity(self,labels,confidence):
      return -1
    
    def generate_heat_maps(self,features,heatmap_type:str):
      '''
      heatmap_type: cam or grid cam
      '''
      heatmaps= np.zeros((len(CLASSES), 7, 7))
      if heatmap_type=="cam":
        for class_index in range(len(CLASSES)):
            # Last layer Weights for this class
            weights = self.weights[class_index].view(1, -1).to("cpu") # 1, 1024

            heatmap=self.cam_heat_map(weights,features) # 7x7

            heatmaps[class_index]=heatmap


      return heatmaps



          
    def cam_heat_map(self,weights,features):
      # Apply Matrix multiplication to get the heatmap
      # weights: 1024 x features: 1024, 7, 7 -> 1, 7, 7
      heatmap = torch.matmul(weights, features.view(features.size(0), -1)).view(features.size(1), features.size(2)) # 7, 7
      heatmap = heatmap.cpu().data.numpy() #torch.Size([7, 7])

      # Apply relu
      heatmap = np.maximum(heatmap, 0)

      # Apply normalization
      if(np.max(heatmap)!=0):
          heatmap = heatmap / np.max(heatmap)

      return heatmap

            

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
    image, heatmaps,labels,confidence,severity,template_based_report=inference.infer(image_path,heatmap_type="cam")
    
    print("image",image.shape)
    print("image_heatmaps",image.shape)
    print("Labels:",labels)
    print("confidence",confidence)
    print("severity",severity)
    print("Report:",template_based_report)



    # Save image using OpenCV
    cv2.imwrite(f'original_224_224.png', image)
          
    
    
    
# python -m src.inference.main "./datasets/images/00000001_000.png"
# python -m src.inference.main "datasets\mimic-cxr-jpg\files\p11\p11001469\s54076811\d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"