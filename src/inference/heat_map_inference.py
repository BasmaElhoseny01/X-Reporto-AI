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

# Constants
ANATOMICAL_REGIONS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6,
    "right hemidiaphragm": 7,
    "left lung": 8,
    "left upper lung zone": 9,
    "left mid lung zone": 10,
    "left lower lung zone": 11,
    "left hilar structures": 12,
    "left apical zone": 13,
    "left costophrenic angle": 14,
    "left hemidiaphragm": 15,
    "trachea": 16,
    "spine": 17,
    "right clavicle": 18,
    "left clavicle": 19,
    "aortic arch": 20,
    "mediastinum": 21,
    "upper mediastinum": 22,
    "svc": 23,
    "cardiac silhouette": 24,
    "cavoatrial junction": 25,
    "right atrium": 26,
    "carina": 27,
    "abdomen": 28
}
REGION_LABELS = {v: k for k, v in ANATOMICAL_REGIONS.items()}



class HeatMapInference:
    heat_map_model=None
    optimal_thresholds=None
    weights=None
    resize_and_pad_transform=None
    normalize_transform=None
    # heat_map_transform=None
    def __init__(self):

        if HeatMapInference.heat_map_model is not None:
            return
        # Create the model
        HeatMapInference.heat_map_model=HeatMap()

        # Load the model
        HeatMapInference.heat_map_model.load_state_dict(torch.load('models/heat_map_best.pth'))

        # Move to Device
        HeatMapInference.heat_map_model.to(DEVICE)
        
        # Evaluation Mode
        HeatMapInference.heat_map_model.eval()

        # Optimal Thrsholds
        HeatMapInference.optimal_thresholds=self.heat_map_model.optimal_thresholds

        # Weights for Last Layer
        HeatMapInference.weights = list(self.heat_map_model.model.classifier.parameters())[-2] #torch.Size([8, 1024])


        HeatMapInference.resize_and_pad_transform = A.Compose([
            A.LongestMaxSize(max_size=HEAT_MAP_IMAGE_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=HEAT_MAP_IMAGE_SIZE, min_width=HEAT_MAP_IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)
        ])

        HeatMapInference.normalize_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0)
        ])


        # HeatMapInference.heat_map_transform=A.Compose([
        #     A.LongestMaxSize(max_size=HEAT_MAP_IMAGE_SIZE, interpolation=cv2.INTER_AREA),

        #     A.PadIfNeeded(min_height=HEAT_MAP_IMAGE_SIZE, min_width=HEAT_MAP_IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT,value=0),
    
        #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
        #     ToTensorV2(p=1.0)
        # ])

    def read_image(self, img_path):
        # Getting image path  with parent path of current folder + image path
        img_path = os.path.join(os.getcwd(), img_path)

        # Fix Image Path
        img_path = img_path.replace("\\", "/")

        # Read Image
        image = cv2.imread(img_path,cv2.IMREAD_COLOR)
        assert image is not None, f"Image at {img_path} is None"

        return image
    
    def _load_and_process_image(self, img_path: str):
        """
        Loads and processes an image from the specified path.

        Args:
            img_path (str): Path to the image file.

        Returns:
            tuple: Original image and processed image tensor.
        """
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
        # image=HeatMapInference.heat_map_transform(image=image)["image"]
        resized_and_padded_image=HeatMapInference.resize_and_pad_transform(image=image)["image"]
        image=HeatMapInference.normalize_transform(image=resized_and_padded_image)["image"]
        
        # Add batch dimension
        image = image.unsqueeze(0)

        return image_org,image
 
    def infer(self,image_path,bounding_boxes,detected_classes,heatmap_type:str = "cam"):
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
            _,y_scores,features=HeatMapInference.heat_map_model(image)

            # Results
            image = image.to("cpu") # Won't Be used :D
            confidence = y_scores.to("cpu").squeeze(0).tolist()
            features=features.to("cpu").squeeze(0)

            labels=[]
            for i,class_confidence in enumerate(confidence):
              label=1*(class_confidence>HeatMapInference.optimal_thresholds[i])
              labels.append(label)

            # Generate Heat Maps
            heat_maps=self.generate_heat_maps(features=features,heatmap_type=heatmap_type)
            
            # Compute Severity
            severity=self.compute_severity(labels=labels,confidence=confidence)

            # Generate generate_template_based_report
            report=self.generate_template_based_report(labels=labels,confidence=confidence,bounding_boxes=bounding_boxes,regions=detected_classes,heat_maps=heat_maps)

            return heat_maps,labels,confidence,severity,report


    def generate_template_based_report(self,labels,confidence,bounding_boxes,regions,heat_maps):   
        """
        Generates a report based on labels and confidence scores.

        Args:
            labels (list): A list of binary labels (1 for positive, 0 for negative) indicating the presence of conditions.
            confidence (list): A list of confidence scores corresponding to the labels.

        Returns:
            list: A list of strings containing the report statements.
        """
        print("Generate Template Based Report")
        report=[]
        # TODO Add Info About the Grey Area
        # template_positive = "The patient has {condition}."
        # template_negative = "The patient does not have {condition}."


        # template_positive = "The patient has {condition} with a confidence of {confidence:.2f}%. " \
        #                     "The findings are primarily located in the {region}. " \
        #                     "Severity: {severity}."
        
        template_positive = "The patient has {condition} with a confidence of {confidence:.2f}%. " \
                            "The findings are primarily located in the {region}. "
        template_negative = "The patient does not have {condition} with a confidence of {confidence:.2f}%. "

        for i, label in enumerate(labels):
            confidence_percent = confidence[i] * 100
            if label == 1:
                # severity_level, severity_score = self.heat_map_severity(heat_maps[i])
                region = self.heatmap_region(heat_maps[i],regions=regions,region_boxes=bounding_boxes)  # Assuming you have a method for this
                report.append(template_positive.format(condition=CLASSES[i],
                                                      confidence=confidence_percent,
                                                      region=region))
                                                    #   severity=severity_level))
            else:
                report.append(template_negative.format(condition=CLASSES[i], confidence=confidence_percent))

        # make report as a single string with new lines
        report = "\n".join(report)
        return report

    def compute_severity(self, labels, confidence):
        
        severity = 0
        
        # calculate severity based on the confidence and labels
        for i, label in enumerate(labels):
            # if the label is positive (1), increase the severity by the confidence score
            if label == 1 and i != 4 and i != 7:  # Skip "No Finding" class
                severity += (confidence[i] - HeatMapInference.optimal_thresholds[i]) /(1 - HeatMapInference.optimal_thresholds[i]) 
    
        print("Severity",severity)
        return severity
        
        # # Weights per findings
        # importance_scores = {
        #     'Atelectasis': 0.2,
        #     'Cardiomegaly': 0.3,
        #     'Edema': 0.1,
        #     'Lung Opacity': 0.05,
        #     'No Finding': 0.05, #**
        #     'Pleural Effusion': 0.15,
        #     'Pneumonia': 0.08,
        #     'Support Devices': 0.07, #**
        # }

        # severity_weights = []

        # # Score = sum(Confidence * heat_map_Severity*weight) #[0-1]

        # # Compute severity score based on heatmap intensity and confidence
        # for i, label in enumerate(labels):
        #     # Calculate severity score based on heatmap intensity (you can adjust this part)
        #     _, severity_score = self.heat_map_severity(heatmaps[i])

        #     # Multiply by confidence score to weigh the severity
        #     weighted_severity = severity_score * confidence[i]

        #     # Use class importance to weight the severity further
        #     importance = importance_scores.get(CLASSES[i], 0.0)
        #     weighted_severity *= importance

        #     # Append the weighted severity to the list
        #     severity_weights.append(weighted_severity)

        # # Normalize weights so they sum up to 10
        # total_weight = sum(severity_weights)
        # if total_weight > 0:
        #     normalized_weight = (total_weight / sum(importance_scores.values())) * 10
        # else:
        #     normalized_weight = 0

        # return normalized_weight
        
    def generate_heat_maps(self,features,heatmap_type:str):
        """
        Generates heat maps based on the given image and features.

        Args:
            image (numpy.ndarray): The original image array.
            features (torch.Tensor): The features extracted from the model.
            heatmap_type (str): Type of heat map to generate ('cam' or 'grad-cam').

        Returns:
            tuple: A tuple containing:
                - heatmaps (numpy.ndarray): Array of heat maps for each class.
                - image_resized (numpy.ndarray): Resized original image.
                - heatmap_resized_plts (numpy.ndarray): Resized heat maps for plotting.
                - blended_images (numpy.ndarray): Blended images with heat maps overlaid.
        """
        heatmaps= np.zeros((len(CLASSES), 7, 7))
        if heatmap_type=="cam":
            for class_index in range(len(CLASSES)):
                # Last layer Weights for this class
                weights = HeatMapInference.weights[class_index].view(1, -1).to("cpu") # 1, 1024

                heatmap=self.cam_heat_map(weights,features) # 7x7

                heatmaps[class_index]=heatmap

        elif heatmap_type=="grad-cam":
            print("grid-cam (Not Implemented)")
            # We need to perform another forward pass to compute the grad
            # Forward Pass
        return heatmaps
          
    def cam_heat_map(self,weights,features):
        """
        Generates Class Activation Map (CAM) heat map.

        Args:
            weights (torch.Tensor): Weights for the class. Shape should be [1, 1024].
            features (torch.Tensor): Features extracted from the model. Shape should be [1024, 7, 7].

        Returns:
            numpy.ndarray: CAM heat map of shape [7, 7].
        """
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
    

    # def grad_cam_heat_map(self):
    #   '''
    #   weights:[1, 1024]
    #   features:[1024, 7, 7]

    #   y_scores: [1] [0-1]

    #   heatmap:[7,7]
    #   '''  
    #   # To be completed  https://chatgpt.com/share/f5cd2d35-d96f-48ba-8807-91809eca1ca6
    #   # print("y_score",y_score)
    #   # # Backward pass to get gradients of target class with respect to the feature maps
    #   # self.heat_map_model.zero_grad()
    #   # y_score.backward(retain_graph=True)
    #   pass


    def project_heat_maps(self,image_path,heat_maps):
        # Read Image (BGR)(original image of shape (2544, 3056, 3) with pixel values in the range [0, 255].
        image=self.read_image(image_path)


        # Initialize Arrays
        heatmap_resized_plts=np.zeros((len(heat_maps), 224, 224,3))
        blended_images=np.zeros((len(heat_maps), 224, 224,3))

        for heatmap_idx,heat_map in enumerate(heat_maps):
            image_resized,heatmap_resized,blended_image = self.project_heat_map(image=image,heat_map=heat_map)
        
            heatmap_resized_plts[heatmap_idx]=heatmap_resized
            blended_images[heatmap_idx]=blended_image

        return image_resized,heatmap_resized_plts,blended_images
      

    def project_heat_map(self,image,heat_map):
        '''
        Overlays a heatmap [] onto an image.

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
        # image_resized = cv2.resize(image, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE)) #(224, 224, 3)
        image_resized=HeatMapInference.resize_and_pad_transform(image=image)["image"]

        # Resize Heat Map to be same size as the image (224x224x3)
        heatmap_resized = cv2.resize(heat_map, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE))

        # Define Color Map [generates a heatmap image from the input cam data, where different intensity values in cam are mapped to corresponding colors in the "jet" colormap.]
        heatmap_resized=cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET) 

        # Weighted Sum 1*img + 0.25*heatmap (224x224x3)
        blended_image = cv2.addWeighted(image_resized,1,heatmap_resized,0.35,0)

        # cv2.imwrite('./img.png',image_resized)
        # cv2.imwrite('./heat.png',heatmap_resized)
        # cv2.imwrite('./blend.png',blended_image)

        return image_resized,heatmap_resized,blended_image


    def heatmap_region(self, heatmap,regions,region_boxes):
        """
        Determines the primary region of activation in the heatmap.

        Args:
        heatmap (numpy.ndarray): The heatmap array of shape (7, 7).
        - regions: list of regions
        - boxes (List[List[float]]): List of bounding boxes.
            Format: [N, 4] => N times [xmin, ymin, xmax, ymax].

        Returns:
            str: Description of the primary region.
        """
        # Resize the heatmap to the original image size
        # #FIX Basma
        heatmap_resized = cv2.resize(heatmap, (HEAT_MAP_IMAGE_SIZE, HEAT_MAP_IMAGE_SIZE))

        # Define Color Map [generates a heatmap image from the input cam data, where different intensity values in cam are mapped to corresponding colors in the "jet" colormap.]
        heatmap_resized=cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET) 
        
        # Average the activation values across the three color channels
        heatmap_gray = np.mean(heatmap_resized, axis=2)
        print("heatmap_gray",heatmap_gray.shape)

        # Resize BB
        new_boxes=[]
        # copy the boxes
        for box in region_boxes:
            new_boxes.append(box.copy())

        for i,box in enumerate(new_boxes):
            # print("Region Box:",box)
            new_boxes[i]=list(map(lambda x: int(x*224/512),box))
            # print("Region Box:",region_boxes[i])
        
        max_activation = 0
        primary_region = None
        


        for region, (x1, y1, x2, y2) in zip(regions,new_boxes):
            # print("Region:",region)
            # print("x1,y1,x2,y2",x1,y1,x2,y2)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            region_activation = heatmap_gray[y1:y2, x1:x2].mean()
            if region_activation > max_activation:
                max_activation = region_activation
                primary_region = region
        return REGION_LABELS[primary_region]

    
    def heat_map_severity(self,heatmap, significant_threshold=0.7, mild_threshold=0.3):
        """
        Determines the severity based on the intensity of the heatmap.

        Args:
            heatmap (numpy.ndarray): The heatmap of activations.
            significant_threshold (float, optional): Threshold for significant finding. Defaults to 0.7.
            mild_threshold (float, optional): Threshold for mild indication. Defaults to 0.3.

        Returns:
            tuple: A tuple containing severity level as string and severity score as int.
                  Severity level can be "significant finding", "mild indication", or "no significant finding".
                  Severity score is 2 for significant finding, 1 for mild indication, and 0 for no significant finding.
        """
        # Convert BGR to grayscale intensity for severity determination
        # Average the activation values across the three color channels [LOSS COLOR INFO]
        # heatmap_gray = np.mean(heatmap, axis=2)  

        # Weighted average to preserve color information
        weights = np.array([0.114, 0.587, 0.299])  # RGB channel weights for grayscale conversion
        heatmap_gray = np.dot(heatmap[..., :3], weights)
    
        max_value = np.max(heatmap_gray)

        if max_value >= significant_threshold * 255:  # Scale threshold to match grayscale range [0, 255]
            return "significant finding", 2
        elif max_value >= mild_threshold * 255:  # Scale threshold to match grayscale range [0, 255]
            return "mild indication", 1
        else:
            return "no significant finding", 0


            

if __name__=="__main__":
    # Check Operation Mode
    if OperationMode.INFERENCE.value!=OPERATION_MODE :
        raise Exception("Operation Mode is not Inference Mode")
    
    # Take image path from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--save_path', type=str, default=None, help='Optional path to save heat maps')


    args = parser.parse_args()
    image_path = args.image_path
    save_path = args.save_path
    
    # Initialize the Inference class
    inference = HeatMapInference()
    print("DEVICE:",DEVICE)
    print("Basma")

    # --------------------------------------------------------------------------------Inference--------------------------------------------------------------------------------
    # Inference
    # bounding_boxes=[[349, 80, 1570, 1591], [627, 245, 1514, 750], [600, 750, 1459, 900], [559, 900, 1459, 1159], [1023, 709, 1487, 927], [709, 191, 1514, 586], [422, 968, 695, 1241], [559, 900, 1678, 1214], [1623, 204, 2660, 1337], [1691, 259, 2564, 750], [1623, 750, 2633, 955], [1664, 955, 2660, 1337], [1623, 709, 2128, 995], [1719, 204, 2510, 586], [2428, 1173, 2701, 1446], [1623, 968, 2660, 1337], [1323, 136, 1705, 873], [1459, 0, 1787, 2544], [736, 27, 1418, 600], [1787, 109, 2564, 545], [1623, 586, 1869, 750], [995, 276, 2431, 1550], [1227, 422, 1991, 804], [1227, 586, 1623, 804], [1118, 818, 2128, 1214], [1118, 818, 1609, 941], [1118, 941, 1609, 1214], [1459, 750, 1582, 873], [559, 968,2660,2544]]
    bounding_boxes=[[0, 0, 255, 255]]
    # detected_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26,27,28,29]
    detected_classes=[1]
    heat_maps,labels,confidence,severity,template_based_report=inference.infer(image_path,bounding_boxes=bounding_boxes,detected_classes=detected_classes,heatmap_type="cam")

    print("Labels:",labels)
    print("confidence",confidence)
    print("heat_maps",heat_maps.shape)

    print("Severity",severity)
    print("Report:",template_based_report)


    # Save Heat Maps (7x7)
    if save_path:
        for i,heat_map in enumerate(heat_maps):
            cv2.imwrite(f"{save_path}/heat_map_{CLASSES[i]}.png",heat_map)
            print(f"Saved Heat Map for {CLASSES[i]} at {save_path}/heat_map_{CLASSES[i]}.png")


    # --------------------------------------------------------------------------------Project Heat Maps--------------------------------------------------------------------------------
    # Project Heat Map on the Original Image
    image_resized,heatmap_resized_plts,blended_images=inference.project_heat_maps(image_path,heat_maps)
    print("image_resized",image_resized.shape)
    print("heatmap_resized_plts",heatmap_resized_plts.shape)
    print("blended_images",blended_images.shape)

    # Save resized image
    if save_path:
        cv2.imwrite(f"{save_path}/image_resized.png",image_resized)
        print(f"Saved Resized Image at {save_path}/image_resized.png")

    # Save Heat Maps & Blended Images
    if save_path:
        for i,heatmap_resized_plt,blended_image in enumerate(zip(heatmap_resized_plts,blended_images)):
            cv2.imwrite(f"{save_path}/heatmap_resized_plt_{CLASSES[i]}.png",heatmap_resized_plt)
            print(f"Saved Heat Map for {CLASSES[i]} at {save_path}/heatmap_resized_plt_{CLASSES[i]}.png")

            cv2.imwrite(f"{save_path}/blended_image_{CLASSES[i]}.png",blended_image)
            print(f"Saved Blended Image for {CLASSES[i]} at {save_path}/blended_image_{CLASSES[i]}.png")

    # ---------------------------------------------------------------------Locate Heat Map Region--------------------------------------------------------------------------------
    # # Locate Heat Map Region
    # for i,heat_map in enumerate(heat_maps):
    #     region=inference.heatmap_region(heatmap=heat_map,regions=None,region_boxes=None)
    #     print(f"Region for {CLASSES[i]}:",region)

    # # Compute Severity
    # severity=inference.compute_severity(labels=labels,confidence=confidence)
    # print("Severity",severity)

    # # Generate Template Based Report
    # template_based_report=inference.generate_template_based_report(labels=labels,confidence=confidence)
    # print("Report:",template_based_report)


    # # Generate Template Based Report & Heat Map
    # # image, heatmap_result,labels,confidence,severity,template_based_report=inference.infer(image_path,heatmap_type="cam")
    # # image, heatmap_result,labels,confidence,severity,template_based_report=inference.infer(image_path,heatmap_type="grad-cam")
    
    # # print("image",image.shape)

    # # print("severity",severity)
    # # print("Report:",template_based_report)

    # # Destruct heat map returns
    # # heatmaps,image_resized,heatmap_resized_plts,blended_images=heatmap_result
    # # print("heatmaps",heatmaps.shape)
    
    
# python -m src.inference.heat_map_inference "./datasets/images/00000001_000.png"
# python -m src.inference.heat_map_inference "datasets\mimic-cxr-jpg\files\p11\p11001469\s54076811\d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"