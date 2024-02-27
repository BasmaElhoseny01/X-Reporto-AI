import torch
import logging
import os
from enum import Enum

# Suppress TensorFlow INFO level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class ModelStage(Enum):
    OBJECT_DETECTOR = 1
    CLASSIFIER = 2
    LANGUAGE_MODEL = 3

class OperationMode(Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3

# device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training / validation / Testing
OPERATION_MODE=3
# Model Stage
MODEL_STAGE=1

# Training Process Parameters
CONTINUE_TRAIN=False# Continue training
TRAIN_RPN=False # Tain only RPN of the object detector
TRAIN_ROI=False # Train only ROI of the object detector
RUN = "1"
EPOCHS=2
BATCH_SIZE=1
# BATCH_SIZE=1
#   TODO: change to 64
EFFECTIVE_BATCH_SIZE = 1
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE//BATCH_SIZE
LM_Batch_Size=1
LEARNING_RATE=1e-3
SCHEDULAR_STEP_SIZE=2 #
SCHEDULAR_GAMMA=0.7
THRESHOLD_LR_SCHEDULER=1e-4
COOLDOWN_LR_SCHEDULER= 1

# Debgging COnfigurations
DEBUG=True

# Modules Configurations:
# Abnormal Binary Classifier Hyper Parameters
ABNORMAL_CLASSIFIER_POS_WEIGHT= 6.0
# Region Selection Classifier Hyper Parameters
REGION_SELECTION_CLASSIFIER_POS_WEIGHT= 2.24

# Pathes to the external files
training_csv_path = 'datasets/train-small.csv'
validation_csv_path = 'datasets/train-small.csv'
test_csv_path = 'datasets/train-small.csv'

# Logging
PERIODIC_LOGGING=False

CHECKPOINT_EVERY_N=300
RECOVER=False
 
SEED=31


def log_config():
    logging.info(f"DEVICE: {DEVICE}")

    logging.info(f"OPERATION_MODE: {OPERATION_MODE}")
    logging.info(f"MODEL_STAGE: {MODEL_STAGE}")

    logging.info(f"CONTINUE_TRAIN: {CONTINUE_TRAIN}")
    logging.info(f"TRAIN_RPN: {TRAIN_RPN}")

    logging.info(f"RUN: {RUN}")
    logging.info(f"EPOCHS: {EPOCHS}")
    logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"EFFECTIVE_BATCH_SIZE: {EFFECTIVE_BATCH_SIZE}")
    logging.info(f"ACCUMULATION_STEPS: {ACCUMULATION_STEPS}")
    logging.info(f"LM_Batch_Size: {LM_Batch_Size}")

    logging.info(f"LEARNING_RATE: {LEARNING_RATE}")
    logging.info(f"SCHEDULAR_STEP_SIZE: {SCHEDULAR_STEP_SIZE}")
    logging.info(f"SCHEDULAR_GAMMA: {SCHEDULAR_GAMMA}")

    logging.info(f"DEBUG: {DEBUG}")
    # logging.info(f"GENERATE_REPORT: {GENERATE_REPORT}")

    logging.info(f"ABNORMAL_CLASSIFIER_POS_WEIGHT: {ABNORMAL_CLASSIFIER_POS_WEIGHT}")
    logging.info(f"REGION_SELECTION_CLASSIFIER_POS_WEIGHT: {REGION_SELECTION_CLASSIFIER_POS_WEIGHT}")

    logging.info(f"training_csv_path: {training_csv_path}")
    logging.info(f"validation_csv_path: {validation_csv_path}")

    logging.info(f"PERIODIC_LOGGING: {PERIODIC_LOGGING}")

    logging.info(f"CHECKPOINT_EVERY_N: {CHECKPOINT_EVERY_N}")
    logging.info(f"RECOVER: {RECOVER}")

def get_config():
    # Get the Configuration dictionary to be saved in check point
    config = {
    "DEVICE": DEVICE,
    "OPERATION_MODE":OPERATION_MODE,
    "MODEL_STAGE": MODEL_STAGE,
    "CONTINUE_TRAIN": CONTINUE_TRAIN,
    "TRAIN_RPN": TRAIN_RPN,
    "RUN": RUN,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
    "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
    "LM_Batch_Size": LM_Batch_Size,
    "LEARNING_RATE": LEARNING_RATE,
    "SCHEDULAR_STEP_SIZE": SCHEDULAR_STEP_SIZE,
    "SCHEDULAR_GAMMA": SCHEDULAR_GAMMA,
    "DEBUG": DEBUG,
    # "GENERATE_REPORT": GENERATE_REPORT,
    "ABNORMAL_CLASSIFIER_POS_WEIGHT": ABNORMAL_CLASSIFIER_POS_WEIGHT,
    "REGION_SELECTION_CLASSIFIER_POS_WEIGHT": REGION_SELECTION_CLASSIFIER_POS_WEIGHT,
    "training_csv_path": training_csv_path,
    "validation_csv_path": validation_csv_path,
    "PERIODIC_LOGGING": PERIODIC_LOGGING,
    "CHECKPOINT_EVERY_N":CHECKPOINT_EVERY_N,
    }

    return config

# By Command Line TODO fix with new arguments :D
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