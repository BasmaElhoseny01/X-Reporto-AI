import torch
import logging
import os
from enum import Enum

# Suppress TensorFlow INFO level logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Suppress Plt INFO level logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class ModelStage(Enum):
    OBJECT_DETECTOR = 1
    CLASSIFIER = 2
    LANGUAGE_MODEL = 3

class OperationMode(Enum):
    TRAINING = 1
    VALIDATION = 2
    EVALUATION = 3
    INFERENCE= 4
    TESTING = 5 #[TODO: Remove this]


############################################################# Global Configuration ############################################################
# device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training / validation / EVALUATION / Inference/ Testing / 
OPERATION_MODE=3
MODEL_STAGE=3

SEED=31

############################################################# Data Configurations ############################################################
# Paths to the external files
training_csv_path = 'datasets/train.csv'
validation_csv_path = 'datasets/valid.csv'
evaluation_csv_path = 'datasets/valid.csv'
test_csv_path:str = 'datasets/test.csv'

heat_map_training_csv_path:str = 'datasets/heat_map_train.csv'
heat_map_validating_csv_path:str = 'datasets/heat_map_val.csv'
heat_map_evaluation_csv_path = 'datasets/heat_map_test.csv'

#############################################################Training Process Parameters############################################################
CONTINUE_TRAIN=False # Continue training
RECOVER=False # Recover from the last checkpoint
TRAIN_RPN=False # Tain only RPN of the object detector
TRAIN_ROI=False # Train only ROI of the object detector

FREEZE_OBJECT_DETECTOR=True
FREEZE = True

RUN = "6"
EPOCHS = 1
BATCH_SIZE = 1
LM_Batch_Size = 29
EFFECTIVE_BATCH_SIZE = 1
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE//BATCH_SIZE

LEARNING_RATE=0.0001

LR_BETA_1=0.9
LR_BETA_2=0.999

Linear_Schedular=True # Linear Schedular if True, Plateau Schedular if False [X-Reporto]
SCHEDULAR_STEP_SIZE=1 # Number of epochs with no improvement after which learning rate will be reduced
SCHEDULAR_GAMMA=0.8 # value multiply lr with
THRESHOLD_LR_SCHEDULER=1e-3 # Threshold for measuring the new optimum, to only focus on significant changes
COOLDOWN_LR_SCHEDULER= 0 # Number of epochs to wait before resuming normal operation after lr has been reduced.

#############################################################  Debugging Configurations ############################################################
CHECKPOINT_EVERY_N=200 # Save checkpoint every N epochs
AVERAGE_EPOCH_LOSS_EVERY=10

# Debugging COnfigurations
DEBUG=True

# Logging
if os.environ.get('GITHUB_REF') == 'refs/heads/main':
    PERIODIC_LOGGING = True  # MUST BE TRUE IN SERVER MODE
else:
    PERIODIC_LOGGING=True

############################################################# Modules Configurations ############################################################
# Weights of each model
OBJECT_DETECTOR_WEIGHT = 1
ABNORMAL_CLASSIFIER_WEIGHT = 2.5
REGION_SELECTION_CLASSIFIER_WEIGHT = 2.5
LM_WEIGHT = 2

# Abnormal Binary Classifier Hyper Parameters
ABNORMAL_CLASSIFIER_POS_WEIGHT= 6.0
# Region Selection Classifier Hyper Parameters
REGION_SELECTION_CLASSIFIER_POS_WEIGHT= 2.24

BERTSCORE_SIMILARITY_THRESHOLD = 0.9 
############################################################# Heat Map Configurations ############################################################
HEAT_MAP_IMAGE_SIZE=224
CLASSES=['Atelectasis', 'Cardiomegaly', 'Edema', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pneumonia', 'Support Devices']

############################################################# Saving Configurations ############################################################
# Check if the script is running from the main branch
if os.environ.get('GITHUB_REF') == 'refs/heads/main':
    SAVE_TO_DRIVE = False  # MUST BE FALSE IN SERVER MODE
else:
    SAVE_TO_DRIVE = False
# SAVE_TO_DRIVE=False # don't change this in server mode
SAVE_IMAGES=False # don't change this in server mode
DRAW_TENSOR_BOARD=100 # Draw Tensor Board every N batches


 

def log_config():
    logging.info(f"DEVICE: {DEVICE}")
    logging.info(f"OPERATION_MODE: {OPERATION_MODE}")
    logging.info(f"MODEL_STAGE: {MODEL_STAGE}")
    logging.info(f"SEED: {SEED}")

    logging.info(f"training_csv_path: {training_csv_path}")
    logging.info(f"validating_csv_path: {validation_csv_path}")
    logging.info(f"evaluation_csv_path: {evaluation_csv_path}")
    logging.info(f"heat_map_training_csv_path {heat_map_training_csv_path}")
    logging.info(f"heat_map_validating_csv_path {heat_map_validating_csv_path}")
    logging.info(f"heat_map_evaluation_csv_path {heat_map_evaluation_csv_path}")

    logging.info(f"CONTINUE_TRAIN: {CONTINUE_TRAIN}")
    logging.info(f"RECOVER: {RECOVER}")
    logging.info(f"TRAIN_RPN: {TRAIN_RPN}")
    logging.info(f"TRAIN_ROI: {TRAIN_ROI}")
    logging.info(f"FREEZE_OBJECT_DETECTOR: {FREEZE_OBJECT_DETECTOR}")
    logging.info(f"FREEZE: {FREEZE}")
    logging.info(f"RUN: {RUN}")
    logging.info(f"EPOCHS: {EPOCHS}")
    logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"LM_Batch_Size: {LM_Batch_Size}")
    logging.info(f"EFFECTIVE_BATCH_SIZE: {EFFECTIVE_BATCH_SIZE}")
    logging.info(f"ACCUMULATION_STEPS: {ACCUMULATION_STEPS}")
    logging.info(f"LEARNING_RATE: {LEARNING_RATE}")
    logging.info(f"LR_BETA_1: {LR_BETA_1}")
    logging.info(f"LR_BETA_2: {LR_BETA_2}")
    logging.info(f"Linear_Schedular: {Linear_Schedular}")
    logging.info(f"SCHEDULAR_STEP_SIZE: {SCHEDULAR_STEP_SIZE}")
    logging.info(f"SCHEDULAR_GAMMA: {SCHEDULAR_GAMMA}")
    logging.info(f"THRESHOLD_LR_SCHEDULER: {THRESHOLD_LR_SCHEDULER}")
    logging.info(f"COOLDOWN_LR_SCHEDULER: {COOLDOWN_LR_SCHEDULER}")

    logging.info(f"CHECKPOINT_EVERY_N: {CHECKPOINT_EVERY_N}")
    logging.info(f"AVERAGE_EPOCH_LOSS_EVERY: {AVERAGE_EPOCH_LOSS_EVERY}")
    logging.info(f"DEBUG: {DEBUG}")
    logging.info(f"PERIODIC_LOGGING: {PERIODIC_LOGGING}")

    logging.info(f"OBJECT_DETECTOR_WEIGHT: {OBJECT_DETECTOR_WEIGHT}")
    logging.info(f"ABNORMAL_CLASSIFIER_WEIGHT: {ABNORMAL_CLASSIFIER_WEIGHT}")
    logging.info(f"REGION_SELECTION_CLASSIFIER_WEIGHT: {REGION_SELECTION_CLASSIFIER_WEIGHT}")
    logging.info(f"LM_WEIGHT: {LM_WEIGHT}")
    logging.info(f"ABNORMAL_CLASSIFIER_POS_WEIGHT: {ABNORMAL_CLASSIFIER_POS_WEIGHT}")
    logging.info(f"REGION_SELECTION_CLASSIFIER_POS_WEIGHT: {REGION_SELECTION_CLASSIFIER_POS_WEIGHT}")
    logging.info(f"BERTSCORE_SIMILARITY_THRESHOLD: {BERTSCORE_SIMILARITY_THRESHOLD}")

    logging.info(f"HEAT_MAP_IMAGE_SIZE: {HEAT_MAP_IMAGE_SIZE}")
    logging.info(f"CLASSES: {CLASSES}")

    logging.info(f"SAVE_TO_DRIVE: {SAVE_TO_DRIVE}")
    logging.info(f"SAVE_IMAGES: {SAVE_IMAGES}")
    logging.info(f"DRAW_TENSOR_BOARD: {DRAW_TENSOR_BOARD}")

    return



def get_config():
    # Get the Configuration dictionary to be saved in check point
    config = {
    "DEVICE": DEVICE,
    "OPERATION_MODE":OPERATION_MODE,
    "MODEL_STAGE": MODEL_STAGE,
    "SEED": SEED,

    "training_csv_path": training_csv_path,
    "validating_csv_path": validation_csv_path,
    "evaluation_csv_path": evaluation_csv_path,
    "heat_map_training_csv_path": heat_map_training_csv_path,
    "heat_map_validating_csv_path": heat_map_validating_csv_path,
    "heat_map_evaluation_csv_path": heat_map_evaluation_csv_path,

    "CONTINUE_TRAIN": CONTINUE_TRAIN,
    "RECOVER": RECOVER,
    "TRAIN_RPN": TRAIN_RPN,
    "TRAIN_ROI": TRAIN_ROI,
    "FREEZE_OBJECT_DETECTOR": FREEZE_OBJECT_DETECTOR,
    "FREEZE": FREEZE,
    "RUN": RUN,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LM_Batch_Size": LM_Batch_Size,
    "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
    "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
    "LEARNING_RATE": LEARNING_RATE,
    "LR_BETA_1": LR_BETA_1,
    "LR_BETA_2": LR_BETA_2,
    "Linear_Schedular": Linear_Schedular,
    "SCHEDULAR_STEP_SIZE": SCHEDULAR_STEP_SIZE,
    "SCHEDULAR_GAMMA": SCHEDULAR_GAMMA,
    "THRESHOLD_LR_SCHEDULER": THRESHOLD_LR_SCHEDULER,
    "COOLDOWN_LR_SCHEDULER": COOLDOWN_LR_SCHEDULER,

    "CHECKPOINT_EVERY_N":CHECKPOINT_EVERY_N,
    "AVERAGE_EPOCH_LOSS_EVERY":AVERAGE_EPOCH_LOSS_EVERY,
    "DEBUG": DEBUG,
    "PERIODIC_LOGGING": PERIODIC_LOGGING,

    "OBJECT_DETECTOR_WEIGHT": OBJECT_DETECTOR_WEIGHT,
    "ABNORMAL_CLASSIFIER_WEIGHT": ABNORMAL_CLASSIFIER_WEIGHT,
    "REGION_SELECTION_CLASSIFIER_WEIGHT": REGION_SELECTION_CLASSIFIER_WEIGHT,
    "LM_WEIGHT": LM_WEIGHT,
    "ABNORMAL_CLASSIFIER_POS_WEIGHT": ABNORMAL_CLASSIFIER_POS_WEIGHT,
    "REGION_SELECTION_CLASSIFIER_POS_WEIGHT": REGION_SELECTION_CLASSIFIER_POS_WEIGHT,
    "BERTSCORE_SIMILARITY_THRESHOLD": BERTSCORE_SIMILARITY_THRESHOLD,

    "HEAT_MAP_IMAGE_SIZE": HEAT_MAP_IMAGE_SIZE,
    "CLASSES": CLASSES,

    "SAVE_TO_DRIVE": SAVE_TO_DRIVE,
    "SAVE_IMAGES": SAVE_IMAGES,
    "DRAW_TENSOR_BOARD": DRAW_TENSOR_BOARD
    }
    return config





# if __name__ == "__main__":
#     config = get_config()
#     print(config)




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
