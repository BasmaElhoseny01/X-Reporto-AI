
EPOCHS =10
BATCH_SIZE=1
IMAGE_SIZE=512
DEPTH=1
LOAD_EPOCH=1000
VERBOSE=True
NUM_THREADS=4
#  n_epochs
N_EPOCHS= 10
ITG=4
ITD=2
LMSE=1
LPERC=4.0
LADV=40
CONTINUE_TRAIN=False
TRAIN_DATA= "datasets/train-200.csv"
TEST_DATA= "datasets/val.csv"
NAME="tomogan"
GPU_IDS=0
CHECKPOINTS_DIR="./checkpoints"
NUM_THREADS=4
PRINT_FREQ=200
LR=1e-4
RESULTS_DIR='/results'
VGG_PATH="src/denoiser/models/vgg19_weights_notop.h5"
LR_POLICY="Linear",

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