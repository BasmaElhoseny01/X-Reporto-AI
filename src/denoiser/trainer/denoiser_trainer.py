
# from src.denoiser.data_loader.custom_dataset_paper import *
import torch
import torch.optim as optim
import sys, os,shutil
import mlflow
from src.denoiser.utils import *
from src.denoiser.models.gan_model import DenoiserGan
from src.denoiser.options.train_option import TrainOptions
from src.denoiser.data_loader.custom_dataset import CustomDataset
from src.denoiser.config import *

class DenoiserTrainer():

    def __init__(self):
        """
        Trainer class for the Denoiser GAN model.

        Attributes:
            arg (TrainOptions): Training options.
            model (DenoiserGan): GAN model for denoising.
            train_dataset (CustomDataset): Training dataset.
            test_dataset (CustomDataset): Testing dataset.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
            schedular_gen (torch.optim.lr_scheduler.StepLR): Learning rate scheduler for generator.
            schedular_dis (torch.optim.lr_scheduler.StepLR): Learning rate scheduler for discriminator.
            itr_out_dir (str): Directory for iteration outputs.
        """
        self.arg=TrainOptions()
        self.model = DenoiserGan(self.arg)
        # load data
        self.train_dataset = CustomDataset(csv_file_path=TRAIN_DATA)
        self.test_dataset = CustomDataset(csv_file_path=EVAL_DATA)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREADS)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREADS)
        # schedulars
        self.schedular_gen = optim.lr_scheduler.StepLR(self.model.optimizer_G, step_size=STEP_SIZE, gamma=GAMMA)
        self.schedular_dis = optim.lr_scheduler.StepLR(self.model.optimizer_D, step_size=STEP_SIZE, gamma=GAMMA)

        self.itr_out_dir = NAME + '-itrOut'
        if os.path.isdir(self.itr_out_dir): 
            shutil.rmtree(self.itr_out_dir)
        os.mkdir(self.itr_out_dir) 
        
        if CONTINUE_TRAIN:
            self.load_model()

        if LOAD_FROM_BATCH:
            self.model.load_every_batch()
    
    def load_model(self):
        """ Load the model."""
        self.model.load_models()
    def save_model(self):
        """ Save the model."""
        self.model.save_models()
    
    def train(self):
        """ Train the model."""
        best_psnt = 0
        best_ssim = 0
        with mlflow.start_run() as run:
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("depth", DEPTH)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("image_size", IMAGE_SIZE)
            mlflow.log_param("generator_iterations", ITG)
            mlflow.log_param("discrimiator_iterations", ITD)
            print('[Info] Start training Epochs ',EPOCHS)
            for epoch in range(EPOCHS):
                print('[Info] Epoch: %i' % (epoch))
                total_epochs_loss = 0
                avg_200_loss = 0

                for batch_idx,(image, lable) in enumerate(self.train_dataloader):
                    X_mb, y_mb = image, lable

                    # loop for generator
                    self.model.set_input((X_mb, y_mb))
                    for _ge in range(ITG):
                        self.model.backward_G()

                    itr_prints_gen = ' Epoch: %i,batch %i, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)' % (\
                    epoch,batch_idx, self.model.loss_G, self.model.loss_G_MSE, self.model.loss_G_GAN, \
                        self.model.loss_G_Perc, )
                    total_epochs_loss += self.model.loss_G
                    avg_200_loss += self.model.loss_G\
                    
                    # loop for discriminator
                    for _de in range(ITD):
                        self.model.backward_D()
                  # print every batch
                    with open("src/denoiser/outputs/iter_logs.txt", "w") as f:
                        print('%s; dloss: %.2f (r%.3f, f%.3f)' % (itr_prints_gen,\
                        self.model.loss_D, self.model.loss_D_real.detach().cpu().numpy().mean(), self.model.loss_D_fake.detach().cpu().numpy().mean(), \
                        ))
                    with open("src/denoiser/outputs/losses.txt", "a") as f:
                        f.write('%s; dloss: %.2f (r%.3f, f%.3f)\n' % (itr_prints_gen,\
                        self.model.loss_D, self.model.loss_D_real.detach().cpu().numpy().mean(), self.model.loss_D_fake.detach().cpu().numpy().mean(), \
                        ))
                
                    # Save model every 200 batches
                    if ((batch_idx+1) % PRINT_FREQ) == 0:
                      self.model.save_every_batch()
                      print('Average loss for last 200 batches is %.2f' % (avg_200_loss/PRINT_FREQ))
                      with open("src/denoiser/outputs/losses.txt", "a") as f:
                        f.write('Average loss for last 200 batches is %.2f\n' % (avg_200_loss/PRINT_FREQ))
                      avg_200_loss = 0
                      # update schedulars
                      self.schedular_gen.step()
                      self.schedular_dis.step()

                with open("src/denoiser/outputs/losses.txt", "a") as f:
                    f.write("average loss for epoch %d is %.2f\n" % (epoch, total_epochs_loss/len(self.train_dataloader)))
                print("average loss for epoch %d is %.2f" % (epoch, total_epochs_loss/len(self.train_dataloader)))

                # evaluate model
                with torch.no_grad():
                    total_ssims = 0
                    total_psnrs = 0
                    for batch_idx,(image, lable) in enumerate(self.test_dataloader):
                        X_test, y_test = image, lable
                        # move to gpu
                        X_test = X_test.to(self.model.device)
                        y_test = y_test.to(self.model.device)
                        # forward
                        self.model.set_input((X_test, y_test))
                        self.model.forward()
                        # get loss
                        pred_img = self.model.fake_C
                        lossMSE = self.model.criterionMSE(pred_img, y_test)
                        lossPerc = self.model.criterionPixel(pred_img, y_test)
                        lossAdv = self.model.criterionGAN(self.model.netD(pred_img), True)
                        lossG = lossMSE*LMSE + lossPerc*LPERC + lossAdv*LADV
                        print('[Info]batch %i Test: gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)' % (batch_idx,lossG, lossMSE, lossAdv, lossPerc))
                        with open("src/denoiser/outputs/losses.txt", "a") as f:
                            f.write('[Info]batch %i Test: gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f)\n' % (batch_idx,lossG, lossMSE, lossAdv, lossPerc))

                         # save images of evaluation  
                        for i in range(0, X_test.shape[0], 1):
                            # move to cpu
                            y_test = y_test.cpu()
                            pred_img = pred_img.cpu()
                            X_test=X_test.cpu()
                            save2image(y_test[i,0,:,:], '%s/gtruth_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                            save2image(X_test[i,DEPTH//2,:,:], '%s/noisy_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                            save2image(pred_img[i,0,:,:].detach().cpu().numpy(), '%s/pred_%d.png' % (self.itr_out_dir,batch_idx*X_test.shape[0]+ i))
                        
                            # calculate metrics
                            mlflow.log_artifacts(self.itr_out_dir)
                            (ssim, psnr) = eval_metrics(y_test[i,0,:,:], pred_img[i,0,:,:].detach().cpu().numpy())
                            total_ssims += ssim
                            total_psnrs += psnr
                     # save best model  
                    if total_ssims/(len(self.test_dataloader)*BATCH_SIZE) > best_ssim :
                        best_ssim = total_ssims/(len(self.test_dataloader)*BATCH_SIZE)
                        self.save_model()
                    if total_psnrs/(len(self.test_dataloader)*BATCH_SIZE) > best_psnt :
                        best_psnt = total_psnrs/(len(self.test_dataloader)*BATCH_SIZE)
                        torch.save(self.model.netG, os.path.join(OUTPUT_DIR, 'netGpsnr.pth'))
                        torch.save(self.model.netD, os.path.join(OUTPUT_DIR, 'netDpsnr.pth'))
                    with open("src/denoiser/outputs/losses.txt", "a") as f:
                        f.write('[Info] Evaluation : AVG SSIM: %.4f, AVG PSNR: %.2f\n' % (total_ssims/(len(self.test_dataloader)*BATCH_SIZE), total_psnrs/(len(self.test_dataloader)*BATCH_SIZE)))
                    print('[Info] Evaluation : AVG SSIM: %.4f, AVG PSNR: %.2f' % (total_ssims/(len(self.test_dataloader)*BATCH_SIZE), total_psnrs/(len(self.test_dataloader)*BATCH_SIZE)))
            # save best model    
            print('[Info] Best PSNR: %.2f' % (best_psnt))
            print('[Info] Best SSIM: %.4f' % (best_ssim))
            with open("src/denoiser/outputs/losses.txt", "a") as f:
                f.write('[Info] Best PSNR: %.2f\n' % (best_psnt))
                f.write('[Info] Best SSIM: %.4f\n' % (best_ssim))
        sys.stdout.flush()

if __name__ == "__main__":
  model= DenoiserTrainer()
  model.train()

  # python -m src.denoiser.trainer.denoiser_trainer