
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
from src.denoiser.options.train_option import TrainOptions
from src.denoiser.data_loader.custom_dataset import CustomDataset


# opt = {
#     "name": NAME,
#     "gpu_ids":0,
#     "checkpoints_dir":CHECKPOINTS_DIR,
#     "batch_size":BATCH_SIZE,
#     "image_size":IMAGE_SIZE,
#     "depth":DEPTH,
#     "load_epoch":LOAD_EPOCH,
#     "verbose":VERBOSE,
#     "lr":LR,
#     "lr_policy":"Linear",
#     "vgg_path":"src\denoiser\models\vgg19_weights_notop.h5",
#     "lmse":LMSE,
#     "lperc":LPERC,
#     "ladv":LADV,
#     "continue_train":CONTINUE_TRAIN,
#     "results_dir":RESULTS_DIR
# }
class DenoiserTrainer():
    def __init__(self):
        self.arg=TrainOptions()
        self.model = TomoGAN(self.arg)
        # self.data_genrator = bkgdGen(data_generator=gen_train_batch_bg(data_file_h5="datasets/train_noise.h5", mb_size=BATCH_SIZE, in_depth=DEPTH, img_size=IMAGE_SIZE), max_prefetch=16)
        self.train_dataset = CustomDataset(csv_file_path=TRAIN_DATA)
        self.test_dataset = CustomDataset(csv_file_path=TEST_DATA)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREADS)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)
        
        self.itr_out_dir = NAME + '-itrOut'
        if os.path.isdir(self.itr_out_dir): 
            shutil.rmtree(self.itr_out_dir)
        os.mkdir(self.itr_out_dir) 

    def train(self):
        with mlflow.start_run() as run:
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("depth", DEPTH)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("image_size", IMAGE_SIZE)
            mlflow.log_param("generator_iterations", ITG)
            mlflow.log_param("discrimiator_iterations", ITD)
            print('[Info] Start training')
            for epoch in range(EPOCHS):
                print('[Info] Epoch: %i' % (epoch))
                total_epochs_loss = 0
              #TODO:
              # make normal dataloader as its better 
              # it make sure to loop on all trainning data 
              # make generator train many times and then train one time discriminator one time all on same example (batch) 

                for batch_idx,(image, lable) in enumerate(self.train_dataloader):
                    X_mb, y_mb = image, lable
                    for _ge in range(ITG):
                        self.model.set_input((X_mb, y_mb))
                        self.model.backward_G()

                    itr_prints_gen = ' Epoch: %i,batch %i, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)' % (\
                     epoch,batch_idx, self.model.loss_G, self.model.loss_G_MSE, self.model.loss_G_GAN, \
                         self.model.loss_G_Perc, )
                    total_epochs_loss += self.model.loss_G

                    for de in range(ITD):
                        self.model.set_input((X_mb, y_mb))
                        self.model.backward_D()
                    
                    with open("src/denoiser/outputs/iter_logs.txt", "w") as f:
                        print('%s; dloss: %.2f (r%.3f, f%.3f)' % (itr_prints_gen,\
                        self.model.loss_D, self.model.loss_D_real.detach().cpu().numpy().mean(), self.model.loss_D_fake.detach().cpu().numpy().mean(), \
                        ))
              
                print("average loss for epoch %d is %.2f" % (epoch, total_epochs_loss/len(self.train_dataloader)))

                # evaluate after finish one epoch
                with torch.no_grad():
                    # X_test, y_test = get1batch4test(data_file_h5=TEST_DATA, in_depth=DEPTH)
                    total_ssims = 0
                    total_psnrs = 0
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
                        lossG = lossMSE*LMSE + lossPerc*LPERC + lossAdv*LADV
                        print('[Info]batch %i Test: gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)' % (batch_idx,lossG, lossMSE, lossAdv, lossPerc))
                        
                        for i in range(0, X_test.shape[0], 1):
                            # move to cpu
                            y_test = y_test.cpu()
                            pred_img = pred_img.cpu()
                            X_test=X_test.cpu()

                            save2image(y_test[i,0,:,:], '%s/gtruth_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                            save2image(X_test[i,DEPTH//2,:,:], '%s/noisy_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                            save2image(pred_img[i,0,:,:].detach().cpu().numpy(), '%s/pred_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                        # if epoch == 0: 
                        #     save2image(y_test[0,0,:,:], '%s/gtruth.png' % (self.itr_out_dir))
                        #     save2image(X_test[0,DEPTH//2,:,:], '%s/noisy.png' % (self.itr_out_dir))
                        

                            mlflow.log_artifacts(self.itr_out_dir)
                            (ssim, psnr) = eval_metrics(y_test[i,0,:,:], pred_img[i,0,:,:].detach().cpu().numpy())
                            total_ssims += ssim
                            total_psnrs += psnr
                            mlflow.log_metric("ssim", ssim)
                        mlflow.log_metric("psnr", psnr)
                    print('[Info] Test: AVG SSIM: %.4f, AVG PSNR: %.2f' % (total_ssims/(len(self.test_dataloader)*BATCH_SIZE), total_psnrs/(len(self.test_dataloader)*BATCH_SIZE)))

        sys.stdout.flush()

if __name__ == "__main__":
  model= DenoiserTrainer()
  model.train()

  # python -m src.denoiser.trainer.denoiser_trainer