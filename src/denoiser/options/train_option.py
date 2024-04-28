from src.denoiser.options.base_option import BaseOptions
import yaml
from src.denoiser.config import*
class TrainOptions(BaseOptions):

    def __init__(self):
        BaseOptions.__init__(self)
        self.initialize_train()

    def initialize_train(self):
        self.print_freq = PRINT_FREQ
        self.n_epochs = N_EPOCHS
        self.vgg_path = config['training']['vgg_path']
        self.lr = float(config['training']['lr'])
        self.itg = int(config['training']['itg'])
        self.itd = int(config['training']['itd'])
        self.lmse = float(config['training']['lmse'])
        self.lperc = float(config['training']['lperc'])
        self.ladv = float(config['training']['ladv'])
        self.xtrain = config['dataset']['xtrain']
        self.ytrain = config['dataset']['ytrain']
        self.xtest = config['dataset']['xtest']
        self.ytest = config['dataset']['ytest']
        self.lr_policy = config['training']['lr_policy']
        self.continue_train = config['training']['continue_train']   
        self.isTrain = True