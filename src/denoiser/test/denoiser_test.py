
# from src.denoiser.data_loader.custom_dataset_paper import *
from src.denoiser.config import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, os, time, argparse, shutil, scipy, h5py, glob
import torchvision.models as models
import mlflow
from src.denoiser.utils import *
from src.denoiser.models.gan_model import TomoGAN
from src.denoiser.options.test_options import TestOptions
from src.denoiser.data_loader.custom_dataset import CustomDataset
from src.denoiser.config import *

class DenoiserTrainer():
    def __init__(self):
        self.arg=TestOptions()
        self.model = TomoGAN(self.arg)
        # self.data_genrator = bkgdGen(data_generator=gen_train_batch_bg(data_file_h5="datasets/train_noise.h5", mb_size=BATCH_SIZE, in_depth=DEPTH, img_size=IMAGE_SIZE), max_prefetch=16)
        self.test_dataset = CustomDataset(csv_file_path=TEST_DATA)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)
        
        self.itr_out_dir = NAME + '-itrOut'
        if os.path.isdir(self.itr_out_dir): 
            shutil.rmtree(self.itr_out_dir)
        os.mkdir(self.itr_out_dir) 
        
        # load models 
        self.load_model()
    
    def load_model(self):
        self.model.load_models()
    def save_model(self):
        self.model.save_models()

    def test(self):
        with mlflow.start_run() as run:
          mlflow.log_param("batch_size", BATCH_SIZE)
          mlflow.log_param("depth", DEPTH)
          mlflow.log_param("epochs", EPOCHS)
          mlflow.log_param("image_size", IMAGE_SIZE)
          mlflow.log_param("generator_iterations", ITG)
          mlflow.log_param("discrimiator_iterations", ITD)
          print('[Info] Start testing')
          with torch.no_grad():
                  total_ssims = 0
                  total_psnrs = 0
                  total_medical_loss=0
                  for batch_idx,(image, lable) in enumerate(self.test_dataloader):
                      X_test, y_test = image, lable
                      # move to gpu
                      X_test = X_test.to(self.model.device)
                      y_test = y_test.to(self.model.device)
                      self.model.set_input((X_test, y_test))
                      self.model.forward()
                      pred_img = self.model.fake_C
                      lossMSE = self.model.criterionMSE(pred_img, y_test)
                      lossPerc = self.model.criterionPixel(pred_img, y_test)
                      lossAdv = self.model.criterionGAN(self.model.netD(pred_img), True)
                      
                      total_medical_loss += lossPerc

                      lossG = lossMSE*LMSE + lossPerc*LPERC + lossAdv*LADV
                      print('[Info]batch %i Test: gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)' % (batch_idx,lossG, lossMSE, lossAdv, lossPerc))
                      for i in range(0, X_test.shape[0], 1):
                          # move to cpu
                          y_test = y_test.cpu()
                          pred_img = pred_img.cpu()
                          X_test=X_test.cpu()

                        #   save2image(y_test[i,0,:,:], '%s/gtruth_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                        #   save2image(X_test[i,DEPTH//2,:,:], '%s/noisy_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                        #   save2image(pred_img[i,0,:,:].detach().cpu().numpy(), '%s/pred_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))

                          mlflow.log_artifacts(self.itr_out_dir)
                          (ssim, psnr) = eval_metrics(y_test[i,0,:,:], pred_img[i,0,:,:].detach().cpu().numpy())
                          total_ssims += ssim
                          total_psnrs += psnr                      
                  print('[Info] Test: AVG SSIM: %.4f, AVG PSNR: %.2f' % (total_ssims/(len(self.test_dataloader)*BATCH_SIZE), total_psnrs/(len(self.test_dataloader)*BATCH_SIZE)))
                  print('[Info] Test: AVG Medical Loss: %.4f' % (total_medical_loss/(len(self.test_dataloader)*BATCH_SIZE)))
        sys.stdout.flush()

if __name__ == "__main__":
  model= DenoiserTrainer()
  model.test()

# python -m src.denoiser.test.denoiser_test