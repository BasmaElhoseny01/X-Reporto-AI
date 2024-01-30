from enum import Enum
import torch

class ModelStage(Enum):
    OBJECT_DETECTOR = 1
    CLASSIFIER = 2
    LANGUAGE_MODEL = 3

MODEL_STAGE=2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')