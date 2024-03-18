
import argparse
from enum import Enum
import torch

class ModelStage(Enum):
    OBJECT_DETECTOR = 1
    CLASSIFIER = 2
    LANGUAGE_MODEL = 3

MODEL_STAGE=3
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# X-Report Trainer Hyper Parameters
EPOCHS=11
LEARNING_RATE=0.0001
BATCH_SIZE=1
LM_Batch_Size=1
SCHEDULAR_STEP_SIZE=1
SCHEDULAR_GAMMA=0.9999999999
DEBUG=True
GENERATE_REPORT=True

# Training Process Parameters
CONTINUE_TRAIN=False# Continue training
TRAIN_RPN=False # Tain only RPN of the object detector
RUN = "0"

# Abnormal Binary Classifier Hyper Parameters
ABNORMAL_CLASSIFIER_POS_WEIGHT = args.abnormal_pos_weight

# Region Selection Classifier Hyper Parameters
REGION_SELECTION_CLASSIFIER_POS_WEIGHT = args.region_pos_weight

# Device Selection
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# # Print or use the updated configuration
# print("*********************************************************************************************************************************************************************")
# print("Using Global Configuration:")
# print("Model Stage", next(name for name, member in ModelStage.__members__.items() if member.value == MODEL_STAGE))
# print("Debug",DEBUG)

# print("Epochs:", EPOCHS)
# print("Learning Rate:", LEARNING_RATE)
# print("Batch Size:", BATCH_SIZE)
# print("Scheduler Step Size:", SCHEDULAR_STEP_SIZE)
# print("Scheduler Gamma:", SCHEDULAR_GAMMA)

# print("Abnormal Region Pos Weight:", ABNORMAL_CLASSIFIER_POS_WEIGHT)
# print("Region Selection Pos Weight:", REGION_SELECTION_CLASSIFIER_POS_WEIGHT)
# print("*********************************************************************************************************************************************************************")
print("Device:", DEVICE)    

#>>>  python your_script.py --epochs 100 --learning_rate 0.0001 --batch_size 32 --scheduler_step_size 5 --scheduler_gamma 0.95 --debug --abnormal_pos_weight 7.5 --region_pos_weight 3.0




# from enum import Enum
# import torch

# class ModelStage(Enum):
#     OBJECT_DETECTOR = 1
#     CLASSIFIER = 2
#     LANGUAGE_MODEL = 3

# MODEL_STAGE=1
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # X-Report Trainer Hyper Parameters
# EPOCHS=50
# LEARNING_RATE=0.00000001
# BATCH_SIZE=1
# SCHEDULAR_STEP_SIZE=1
# SCHEDULAR_GAMMA=0.9999999999
# DEBUG=True

# # Abnormal Binary Classifier Hyper Parameters
# ABNORMAL_CLASSIFIER_POS_WEIGHT= 6.0

# # Region Selection Classifier Hyper Parameters
# REGION_SELECTION_CLASSIFIER_POS_WEIGHT= 2.2