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
        image_org = cv2.imread(img_path,cv2.IMREAD_COLOR)
        assert image_org is not None, f"Image at {img_path} is None"

        # convert the image from BGR to RGB
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)  #(3056, 2544, 3) [0-255]

        # to float32 
        image=np.array(image).astype("float32")

        # Apply Transformations
        image=self.heat_map_transform(image=image)["image"]
              
        # Add batch dimension
        image = image.unsqueeze(0)


        # # Resize Original Image to be 224*244
        # image_org = cv2.resize(image_org, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE)) #(224, 224, 3)


        return image_org,image
 
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
        image_org,image=self._load_and_process_image(img_path=image_path)

        with torch.no_grad():
            image=image.to(DEVICE)

            # Forward Pass
            _,y_scores,features=self.heat_map_model(image)

            # Results
            image = image.to("cpu") # Won't B used :D
            confidence = y_scores.to("cpu").squeeze(0).tolist()
            features=features.to("cpu").squeeze(0)

            labels=[]
            for i,class_confidence in enumerate(confidence):
              label=1*(class_confidence>self.optimal_thresholds[i])
              labels.append(label)

            heatmaps,image_resized,heatmap_resized_plts,blended_images=self.generate_heat_maps(image=image_org,features=features,heatmap_type=heatmap_type)
            heatmap_result=(heatmaps,image_resized,heatmap_resized_plts,blended_images)

            severity=self.compute_severity(labels,confidence)

            template_based_report=self.generate_template_based_report(labels=labels,confidence=confidence)

            return image_org,heatmap_result,labels,confidence,severity,template_based_report


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
    
    def generate_heat_maps(self,image,features,heatmap_type:str):
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
      

      heatmap_resized_plts=np.zeros((len(CLASSES), 224, 224,3))
      blended_images=np.zeros((len(CLASSES), 224, 224,3))
      # Project Heat Map on the Original Image
      for class_index in range(len(CLASSES)):
        image_resized,heatmap_resized,blended_image=self.project_heat_map(image=image,heatmap=heatmaps[class_index])

        heatmap_resized_plts[class_index]=heatmap_resized
        blended_images[class_index]=blended_image


      return heatmaps,image_resized,heatmap_resized_plts,blended_images



          
    def cam_heat_map(self,weights,features):
      '''
      weights:[1, 1024]
      features:[1024, 7, 7]

      heatmap:[7,7]
      '''
      # Apply Matrix multiplication to get the heatmap
      # weights: 1024 x features: 1024, 7, 7 -> 1, 7, 7
      heatmap = torch.matmul(weights, features.view(features.size(0), -1)).view(features.size(1), features.size(2)) # 7, 7
      heatmap = heatmap.cpu().data.numpy() #torch.Size([7, 7])

      # Apply relu
      heatmap = np.maximum(heatmap, 0)

      # Apply normalization (If the maximum value is zero, normalization would lead to division by zero)
      if(np.max(heatmap)!=0): 
          heatmap = heatmap / np.max(heatmap)

      return heatmap

    def project_heat_map(self,image,heatmap):
      '''
      Overlays a heatmap onto an image.

      Parameters:
      - image: numpy.ndarray
          Original BGR image of shape (2544, 3056, 3) with pixel values in the range [0, 255].
      - heatmap: numpy.ndarray
          Heatmap of shape (7, 7) with normalized values in the range [0, 1].

      Returns:
      - image_resized: numpy.ndarray
          The original image resized to (224, 224, 3) in BGR format.
      - heatmap_resized: numpy.ndarray
          The heatmap resized to (224, 224, 3) in BGR format.
      - blended_image: numpy.ndarray
          The blended image of size (224, 224, 3) in BGR format, which is a weighted sum of the resized image and the heatmap.
      '''
      # Resize Image to be HEAT_MAP_IMAGE_SIZExHEAT_MAP_IMAGE_SIZEx3 (224x224x3)
      image_resized = cv2.resize(image, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE)) #(224, 224, 3)

      # Reize Heat Map to be same size as the image (224x224x3)
      heatmap_resized = cv2.resize(heatmap, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE))

      # Define Color Map [generates a heatmap image from the input cam data, where different intensity values in cam are mapped to corresponding colors in the "jet" colormap.]
      heatmap_resized=cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET) 

      # Weighted Sum 1*img + 0.25*heatmap (224x224x3)
      blended_image = cv2.addWeighted(image_resized,1,heatmap_resized,0.35,0)

    #   cv2.imwrite('./img.png',image_resized)
    #   cv2.imwrite('./heat.png',heatmap_resized)
    #   cv2.imwrite('./blend.png',blended_image)

      return image_resized,heatmap_resized,blended_image




            

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
    image, heatmap_result,labels,confidence,severity,template_based_report=inference.infer(image_path,heatmap_type="cam")
    
    print("image",image.shape)
    print("Labels:",labels)
    print("confidence",confidence)
    print("severity",severity)
    print("Report:",template_based_report)

    # Destruct heat map returns
    heatmaps,image_resized,heatmap_resized_plts,blended_images=heatmap_result
    print("heatmaps",heatmaps.shape)
    print("image_resized",image_resized.shape)
    print("heatmap_resized_plts",heatmap_resized_plts.shape)
    print("blended_images",blended_images.shape)
    
    
# python -m src.inference.heat_map_inference "./datasets/images/00000001_000.png"
# python -m src.inference.heat_map_inference "datasets\mimic-cxr-jpg\files\p11\p11001469\s54076811\d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"