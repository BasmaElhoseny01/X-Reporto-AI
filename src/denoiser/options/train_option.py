from src.denoiser.options.base_option import BaseOptions
from src.denoiser.config import*
# We will define the train options for the denoiser here.
class TrainOptions(BaseOptions):

    def __init__(self):
        BaseOptions.__init__(self)
        self.initialize_train()

    def initialize_train(self):
        self.print_freq = PRINT_FREQ
        self.n_epochs = N_EPOCHS
        self.vgg_path = VGG_PATH
        self.lr = LR
        self.itg = ITG
        self.itd = ITD
        self.lmse = LMSE
        self.lperc = LPERC
        self.ladv = LADV
        self.train_data = TRAIN_DATA
        self.test_data =TEST_DATA
        self.lr_policy = LR_POLICY
        self.continue_train = CONTINUE_TRAIN 
        self.isTrain = True
        