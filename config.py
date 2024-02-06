
# import argparse
# from enum import Enum
# import torch

# class ModelStage(Enum):
#     OBJECT_DETECTOR = 1
#     CLASSIFIER = 2
#     LANGUAGE_MODEL = 3

# # Parse command-line arguments
# parser = argparse.ArgumentParser(description="Configuration")

# parser.add_argument("--model_stage", type=int, choices=[1, 2, 3], default=1, help="Model stage: 1 (OBJECT_DETECTOR), 2 (CLASSIFIER), or 3 (LANGUAGE_MODEL)")
# parser.add_argument("--debug", action="store_true", help="Enable debugging")

# parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
# parser.add_argument("--learning_rate", type=float, default=0.00000001, help="Learning rate")
# parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
# parser.add_argument("--scheduler_step_size", type=int, default=1, help="Scheduler step size")
# parser.add_argument("--scheduler_gamma", type=float, default=0.9999999999, help="Scheduler gamma")

# parser.add_argument("--abnormal_pos_weight", type=float, default=6.0, help="Abnormal classifier positive weight")
# parser.add_argument("--region_pos_weight", type=float, default=2.2, help="Region selection classifier positive weight")

# args = parser.parse_args()

# # Update configuration with command-line arguments
# MODEL_STAGE=args.model_stage
# DEBUG = args.debug

# # Training Parameters
# EPOCHS = args.epochs
# LEARNING_RATE = args.learning_rate
# BATCH_SIZE = args.batch_size
# SCHEDULAR_STEP_SIZE = args.scheduler_step_size
# SCHEDULAR_GAMMA = args.scheduler_gamma

# # Abnormal Binary Classifier Hyper Parameters
# ABNORMAL_CLASSIFIER_POS_WEIGHT = args.abnormal_pos_weight

# # Region Selection Classifier Hyper Parameters
# REGION_SELECTION_CLASSIFIER_POS_WEIGHT = args.region_pos_weight

# # Device Selection
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # # Print or use the updated configuration
# print("*********************************************************************************************************************************************************************")
# print("Using Global Configuration:")
# print("Model Stage", next(name for name, member in ModelStage.__members__.items() if member.value == MODEL_STAGE))
# # print("Debug",DEBUG)

# # print("Epochs:", EPOCHS)
# # print("Learning Rate:", LEARNING_RATE)
# # print("Batch Size:", BATCH_SIZE)
# # print("Scheduler Step Size:", SCHEDULAR_STEP_SIZE)
# # print("Scheduler Gamma:", SCHEDULAR_GAMMA)

# # print("Abnormal Region Pos Weight:", ABNORMAL_CLASSIFIER_POS_WEIGHT)
# # print("Region Selection Pos Weight:", REGION_SELECTION_CLASSIFIER_POS_WEIGHT)
# # print("*********************************************************************************************************************************************************************")


# #>>>  python your_script.py --epochs 100 --learning_rate 0.0001 --batch_size 32 --scheduler_step_size 5 --scheduler_gamma 0.95 --debug --abnormal_pos_weight 7.5 --region_pos_weight 3.0




from ast import Continue
from enum import Enum
import torch

class ModelStage(Enum):
    OBJECT_DETECTOR = 1
    CLASSIFIER = 2
    LANGUAGE_MODEL = 3

MODEL_STAGE=2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# X-Report Trainer Hyper Parameters
EPOCHS=4
LEARNING_RATE=0.00000001
BATCH_SIZE=40
SCHEDULAR_STEP_SIZE=1
SCHEDULAR_GAMMA=0.9999999999
DEBUG=True

# Training Process Parameters
CONTINUE_TRAIN=False # Continue training
TRAIN_RPN=False # Tain only RPN of the object detector
RUN = 0

# Abnormal Binary Classifier Hyper Parameters
ABNORMAL_CLASSIFIER_POS_WEIGHT= 6.0

# Region Selection Classifier Hyper Parameters
REGION_SELECTION_CLASSIFIER_POS_WEIGHT= 2.2