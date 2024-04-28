import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from src.utils import plot_single_image

# Modules
from src.x_reporto.models.x_reporto_v1 import XReportoV1
from transformers import GPT2Tokenizer


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


                               

    def generate_image_report(self,image_path):
        # Read the image
        image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        #print(image.shape)

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
            with open("report.txt", "w") as file:
                # Iterate over each sentence in the list
                for sentence in lm_sentences_decoded:
                    file.write(sentence + "\n")

        
        # Input is Image
        # Output is Image with bounding box
        # Selected Regions / Abnormal Region
        # Report

    #     pass




# from src.object_detector.models.object_detector_factory import ObjectDetector
# from src.binary_classifier.models.binary_classifier_selection_region_factory import BinaryClassifierSelectionRegion
# from src.binary_classifier.models.binary_classifier_region_abnormal_factory import BinaryClassifierRegionAbnormal
# from src.language_model.GPT2.gpt2_model import CustomGPT2
# from src.language_model.GPT2.config import Config
if __name__=="__main__":
    
    # Initialize the Inference class
    inference = Inference()

    # Generate the report
    inference.generate_image_report(image_path='./datasets/mimic-cxr-jpg/files/p10/p10003299/s57344656/f5414268-e553a141-39841839-4f303c85-d94d1190.jpg')

    
    pass    
       
    
    
    # For Debugging
    # Create Object detector
    # object_detector = ObjectDetector().create_model()
    # torch.save(object_detector.state_dict(), "models/object_detector_best.pth")


    #selection_region = BinaryClassifierSelectionRegion().create_model()
    #torch.save(selection_region.state_dict(), "models/region_classifier_best.pth")
    
    # region_abnormal=BinaryClassifierRegionAbnormal().create_model()
    # torch.save(region_abnormal.state_dict(), "models/abnormal_classifier_best.pth")
    
    # lm=CustomGPT2
    # config = Config()
    ## load small gpt2 config
    #config.d_model = 768
    #config.d_ff1 = 768
    #config.d_ff2 = 768
    #config.d_ff3 = 768
    #config.num_heads = 12
    #config.num_layers = 12
    #config.vocab_size = 50257
    #config.pretrained_model = "gpt2"
    #config.max_seq_len = 1024
    #config.ignore_index = -100
    #image_config = Config()
    #image_config.d_model = 1024
    #image_config.d_ff1 = 1024
    #image_config.d_ff2 = 1024
    ## image_config.d_ff2 = 768
    #image_config.d_ff3 = 768
    #image_config.num_heads = 16
    #image_config.num_layers = 24
    #image_config.vocab_size = 50257
    #lm = CustomGPT2(config,image_config)
    #torch.save(lm.state_dict(), "models/LM_best.pth")
    
    
# python -m src.inference.main