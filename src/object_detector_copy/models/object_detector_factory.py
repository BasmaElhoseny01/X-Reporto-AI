from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn


from src.object_detector_copy.models.frccn_object_detector_v1 import FrcnnObjectDetectorV1

class ObjectDetector():
    def __init__(self):
        pass
    
    def create_model(self) -> FrcnnObjectDetectorV1:
        return FrcnnObjectDetectorV1()

# model=ObjectDetector().create_model()
# print(model)