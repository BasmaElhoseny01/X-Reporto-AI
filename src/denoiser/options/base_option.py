# We will define the base options for the denoiser here.
from src.denoiser.config import *
class BaseOptions():

    def __init__(self):
        self.initialize()
    
    def initialize(self):
        self.name =NAME
        self.gpu_ids = GPU_IDS
        self.batch_size =BATCH_SIZE
        self.image_size = IMAGE_SIZE
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.num_threads =NUM_THREADS
        self.depth = DEPTH
        self.load_epoch = LOAD_EPOCH
        self.verbose = VERBOSE
        
    
        