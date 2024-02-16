import torch
import sys


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Trainer Hyper Parameters
EPOCHS=51
LEARNING_RATE=0.001
BATCH_SIZE=1
SCHEDULAR_STEP_SIZE=1
SCHEDULAR_GAMMA=0.9999999999
DEBUG=True

# Training Process Parameters
CONTINUE_TRAIN=True # Continue training
RUN = "0"

# paths to the datasets
Heat_map_train_csv_path:str = 'datasets/HeatMapData.csv'
Heat_map_test_csv_path:str = 'datasets/HeatMapData.csv'
