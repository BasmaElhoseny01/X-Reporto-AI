from enum import Enum
import torch

class ModelStage(Enum):
    OBJECT_DETECTOR = 1
    CLASSIFIER = 2
    LANGUAGE_MODEL = 3

MODEL_STAGE=1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# X-Report Trainer Hyper Parameters
EPOCHS=50
LEARNING_RATE=0.00000001
BATCH_SIZE=1
SCHEDULAR_STEP_SIZE=1
SCHEDULAR_GAMMA=0.9999999999
DEBUG=True

# Abnormal Binary Classifier Hyper Parameters
ABNORMAL_CLASSIFIER_POS_WEIGHT= 6.0

# Region Selection Classifier Hyper Parameters
REGION_SELECTION_CLASSIFIER_POS_WEIGHT= 2.2
