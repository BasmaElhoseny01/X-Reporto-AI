
from src.denoiser.data_loader.custom_dataset_paper import *
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
        self.data_genrator = bkgdGen(data_generator=gen_train_batch_bg(data_file_h5="datasets/train_noise.h5", mb_size=BATCH_SIZE, in_depth=DEPTH, img_size=IMAGE_SIZE), max_prefetch=16)
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
            i=0
            gen=None
            dis=None
            for epoch in range(EPOCHS):
                #TODO:
                # make normal dataloader as its better 
                # it make sure to loop on all trainning data 
                # make generator train many times and then train one time discriminator one time all on same example (batch) 
                time_git_st = time.time()
                print("i = ",i)

                for _ge in range(ITG):
                    X_mb, y_mb = self.data_genrator.next()
                    if i==0:
                      gen=np.copy(X_mb)
                    else:
                      dis=np.copy(X_mb)
                    i+=X_mb.shape[0]
                    self.model.set_input((X_mb, y_mb))
                    self.model.backward_G()

                itr_prints_gen = '[Info] Epoch: %05d, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f), gen_elapse: %.2fs/itr' % (\
                            epoch, self.model.loss_G, self.model.loss_G_MSE*LMSE, self.model.loss_G_GAN*LADV, \
                                self.model.loss_G_Perc*LPERC, (time.time() - time_git_st)/ITG, )
                time_dit_st = time.time()

                if np.all(gen == dis) :
                    print("identical")
                    break
                else:
                    print("nooooooooooooooooo")

                for de in range(ITD):
                    X_mb, y_mb = self.data_genrator.next()
                    dis=np.copy(X_mb)
                    i+=X_mb.shape[0]
                    self.model.set_input((X_mb, y_mb))
                    self.model.backward_D()
                
                if np.all(gen == dis) :
                    print("identical")
                    break
                else:
                    print("nooooooooooooooooo")
                with open("src/denoiser/outputs/iter_logs.txt", "w") as f:
                    print('%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr' % (itr_prints_gen,\
                    self.model.loss_D, self.model.loss_D_real.detach().cpu().numpy().mean(), self.model.loss_D_fake.detach().cpu().numpy().mean(), \
                    (time.time() - time_dit_st)/ITD, time.time()-time_git_st))
            
                if epoch % (200//ITG) == 0:
                    with torch.no_grad():
                        X_test, y_test = get1batch4test(data_file_h5=TEST_DATA, in_depth=DEPTH)
                        self.model.set_input((X_test, y_test))
                        self.model.forward()
                        pred_img = self.model.fake_C
                        save2image(pred_img[0,0,:,:].detach().cpu().numpy(), '%s/it%05d.png' % (self.itr_out_dir, epoch))
                        if epoch == 0: 
                            save2image(y_test[0,0,:,:], '%s/gtruth.png' % (self.itr_out_dir))
                            save2image(X_test[0,DEPTH//2,:,:], '%s/noisy.png' % (self.itr_out_dir))
                            
            mlflow.log_artifacts(self.itr_out_dir)
            (ssim, psnr) = eval_metrics(y_test[0,0,:,:], pred_img[0,0,:,:].detach().cpu().numpy())
            mlflow.log_metric("ssim", ssim)
            mlflow.log_metric("psnr", psnr)

        sys.stdout.flush()

if __name__ == "__main__":
  model= DenoiserTrainer()
  model.train()

  # python -m src.denoiser.trainer.denoiser_trainer