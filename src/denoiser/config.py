
EPOCHS =100
BATCH_SIZE=2
IMAGE_SIZE=512
DEPTH=1
LOAD_EPOCH=1000
VERBOSE=True
NUM_THREADS=4
#  n_epochs
N_EPOCHS= 10
ITG=4
ITD=3
LMSE=1
LPERC=100
LADV=20
# GEN=True

OUTPUT_DIR="src/denoiser/denoiser_models"
CONTINUE_TRAIN=True
TRAIN_DATA= "datasets/train.csv"
TEST_DATA= "datasets/trainX.csv"
EVAL_DATA= "datasets/trainX.csv"
# TEST_DATA= "datasets/train.csv"
NAME="tomogan"
GPU_IDS=0
CHECKPOINTS_DIR="./checkpoints"
NUM_THREADS=4
PRINT_FREQ=200
LR=1e-3
RESULTS_DIR='/results'
VGG_PATH="src/denoiser/models/vgg19_weights_notop.h5"
LR_POLICY="Linear"
RESNET_PATH="src/denoiser/denoiser_models/resnet50.pth"

# training:
#   n_epochs: 1000
#   print_freq: 200
#   lr: 1e-4
#   lr_policy: linear
#   vgg_path: ./vgg19-dcbb9e9d.pth
#   itg: 1
#   itd: 2
#   lmse: 0.5
#   lperc: 2.0
#   ladv: 20
#   continue_train: True

# evaluation:
#   results_dir: ./results
#   num_test: 10