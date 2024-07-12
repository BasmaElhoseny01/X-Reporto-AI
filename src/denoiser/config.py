
# Configurations for the denoiser model
""" Configurations for the denoiser model"""
EPOCHS = 15
BATCH_SIZE=12
IMAGE_SIZE=512
DEPTH=1
LOAD_EPOCH=1000
VERBOSE=True
NUM_THREADS=4
N_EPOCHS= 10
ITG=4
ITD=3
LMSE=1
LPERC=100
LADV=20
PRINT_FREQ=200
STEP_SIZE=50
GAMMA=0.1
LR=1e-4
IMAGE_INPUT_SIZE=512
NUM_THREADS=8
GPU_IDS=0

""" Paths for the denoiser model"""
OUTPUT_DIR="src/denoiser/denoiser_models"
BATCH_OUTPUT_DIR="src/denoiser/denoiser_models/batch_models"
TRAIN_DATA= "datasets/train.csv"
TEST_DATA= "datasets/test.csv"
EVAL_DATA= "datasets/valid.csv"
NAME="denoisergan"
CHECKPOINTS_DIR="./checkpoints"
RESULTS_DIR='/results'
VGG_PATH="src/denoiser/models/vgg19_weights_notop.h5"
LR_POLICY="Linear"

""" Flags for the denoiser model"""
CONTINUE_TRAIN=True
LOAD_FROM_BATCH=False
