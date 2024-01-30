from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn


from src.object_detector_copy.models.frccn_object_detector_v1 import FrcnnObjectDetectorV1
from src.object_detector_copy.models.object_detector_paper.object_detector import ObjectDetectorOriginal


class ObjectDetectorWrapper(nn.Module):
    def __init__(self, object_detector):
        super().__init__()
        self.object_detector = object_detector

    def forward(self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None):
        # Modify Input/Output as the Required by submodule
        if self.training:
            object_detector_losses, object_detector_features, object_detector_labels=self.object_detector(images,targets)
            if(isinstance(self.object_detector, ObjectDetectorOriginal)):
                # Convert boolean tensor to indices tensor
                indices_tensor = object_detector_labels.nonzero()

                # Split indices tensor into a list of tensors based on unique row values
                indices_list = [indices_tensor[indices_tensor[:, 0] == i, 1]+1 for i in range(indices_tensor[:, 0].max() + 1)]

                # Convert the list of tensors to a list of Python lists
                indices_list_of_lists = [indices.tolist() for indices in indices_list]

                # Convert the list of lists to a list of tensors
                object_detector_labels = [torch.tensor(indices, device='cuda:0') for indices in indices_list_of_lists]


            object_detector_boxes=None

        else:
            object_detector_losses,object_detector_boxes,object_detector_features,object_detector_labels=self.object_detector(images,targets)
            object_detector_losses=None

        return object_detector_losses,object_detector_boxes,object_detector_labels,object_detector_features

class ObjectDetector():
    def __init__(self):
        pass
    
    def create_model(self) -> ObjectDetectorWrapper:

        # Add Required Object Detector Module
        
        # object_detector_module=FrcnnObjectDetectorV1()
        object_detector_module=ObjectDetectorOriginal(return_feature_vectors=True)
        
        return ObjectDetectorWrapper(object_detector_module)

# model=ObjectDetector().create_model()
# print(model)
# model.train()
# model(10,20)
