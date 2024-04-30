
EPOCHS =200
BATCH_SIZE=1
IMAGE_SIZE=512
DEPTH=3
LOAD_EPOCH=1000
VERBOSE=True
NUM_THREADS=4
#  n_epochs
N_EPOCHS= 1000
ITG=1
ITD=1
LMSE=0.5
LPERC=2.0
LADV=20
CONTINUE_TRAIN=True
TRAIN_DATA= "./datasets/train_noise.h5"
TEST_DATA= "./datasets/noisy4test.h5"
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