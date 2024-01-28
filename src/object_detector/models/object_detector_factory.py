from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn as nn


from src.object_detector.models.frccn_object_detector_v1 import FrcnnObjectDetectorV1

class ObjectDetector(nn.Module):
    def __init__(self):
        self.model=FrcnnObjectDetectorV1()

    def forward(self,images: Tensor ,targets: Optional[List[Dict[str, Tensor]]] = None):
        self.model(images ,targets)