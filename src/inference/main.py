import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from src.x_reporto.models.x_reporto_v1 import XReportoV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Inference:
    def __init__(self):

        # Read the model
        # self.x_reporto = XReportoV1(object_detector_path="models/object_detector.pth",
        #                        region_classifier_path="models/region_classifier.pth",
        #                        abnormal_classifier_path="models/abnormal_classifier.pth",
        #                        language_model_path="models/language_model.pth")
        print("Model Loaded")


                               

    def generate_image_report(self,image_path):
        # Read the image
        image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        print(image.shape)

        transform = A.Compose([
            A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=0.474, std=0.301),
            ToTensorV2(),
        ])
        image = transform(image=image)['image']


        # Add batch dimension
        image = image.unsqueeze(0)

        # Move the image to GPU
        image = image.to(DEVICE)   

        # Inference Pass
        self.x_reporto(images=image,generate_sentence=False,use_beam_search=False)

        
        # Input is Image
        # Output is Image with bounding box
        # Selected Regions / Abnormal Region
        # Report

    #     pass



from src.object_detector.models.object_detector_factory import ObjectDetector
from utils import save_model
if __name__=="__main__":
    # For Debugging
    # Create Object detector
    object_detector = ObjectDetector().create_model()
    torch.save(object_detector.state_dict(), "models/object_detector.pth")


    # self.binary_classifier_selection_region = BinaryClassifierSelectionRegion().create_model()
    # self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal().create_model()
    # print("Inference")
    # # Initialize the Inference class
    # inference = Inference()


    # inference.generate_image_report(image_path='./datasets/mimic-cxr-jpg/files/p10/p10001884/s50279568/3892f17f-8fa034e8-e9b81865-01c48bbb-b9452626.jpg')

    # Generate the report
    # inference.generate_image_report("path/to/image.jpg")
    pass