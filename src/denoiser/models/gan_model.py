import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
# Custom imports from your project
from config import OUTPUT_DIR,BATCH_OUTPUT_DIR,OPERATION_MODE,OperationMode
from src.denoiser.models.unet_parts import OutConv, DoubleConv, Down, Up
from src.denoiser.models.base_model import BaseModel
from src.denoiser.models.loss_functions import GANLoss, PixelLoss
import torchvision.models as models
from torchvision.models import VGG19_Weights
from src.denoiser.options.train_option import TrainOptions

# Calculate the output size of the feature map
def calculate_feature_output_size(img_size, kernel_size, padding, stride):
    """
    Calculates the output size of a feature map after applying a convolution layer.

    Args:
        img_size (int): Size of the input image.
        kernel_size (int): Size of the convolution kernel.
        padding (int): Padding applied to the input.
        stride (int): Stride of the convolution.

    Returns:
        int: Size of the output feature map.
    """
    return int((img_size - kernel_size + 2*padding)/stride) + 1

 # Define the UNet model
class UNet(nn.Module):
    def __init__(self, channels_in, channels_out):
        """
        Initializes the UNet model for image-to-image translation.

        Args:
            channels_in (int): Number of input channels.
            channels_out (int): Number of output channels.
        """
        super(UNet, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        # Encoder
        self.inc = OutConv(channels_in, out_channels=8, relu=True)
        self.conv1 = DoubleConv(8, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.bottom = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.up01 = Up(512, 256)
        self.up10 = Up(256, 128)
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 32)
        self.conv2 = OutConv(32, 16, relu=True)
        self.conv3 = OutConv(16, channels_out, relu=False)
    # Define the forward pass
    def forward(self, x):
        """
        Forward pass through the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels_in, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels_out, height, width).
        """
        x1 = self.inc(x)
        x2 = self.conv1(x1)
        x3 = self.down1(x2)
        x4 = self.down2(x3)
        x5 = self.down3(x4)
        x6 = self.down4(x5)
        x7 = self.down5(x6)
        x7 = self.bottom(x7)
        x = self.up01(x7, x6)
        x = self.up10(x, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        """
        Forward pass through the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels_in, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels_out, height, width).
        """
        N = calculate_feature_output_size(img_size, 3, 0, 1)
        N = calculate_feature_output_size(N, 3, 0, 2)
        N = calculate_feature_output_size(N, 3, 0, 1)
        N = calculate_feature_output_size(N, 3, 0, 2)
        N = calculate_feature_output_size(N, 3, 0, 1)
        self.feature_size = calculate_feature_output_size(N, 3, 0, 2)
        self.image_to_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=2),
            nn.LeakyReLU()
        )
        self.features_to_score = nn.Sequential(
            nn.Linear(4*self.feature_size*self.feature_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1))
    # Define the forward pass
    def forward(self, x):
        """
        Forward pass through the Discriminator model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Output tensor indicating the likelihood of the input being a real image.
        """
        x = self.image_to_features(x)
        x = x.contiguous().view((x.shape[0], -1))
        x = self.features_to_score(x)
        return x

class DenoiserGan(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        """
        Forward pass through the Discriminator model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Output tensor indicating the likelihood of the input being a real image.
        """
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_B', 'fake_C', 'real_C']
        if self.isTrain :
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = UNet(opt.depth, 1)
        if self.isTrain:
            self.netD = Discriminator(opt.image_size)
            self.netD.to(self.device)
     
        self.netG.to(self.device)
        
        self.itr_out_dir = opt.name + '-itrOut'
        
        self.depth = opt.depth

        if self.isTrain:
            # Get the vgg19 model
            vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)
            vgg19= vgg19.features
            if torch.cuda.is_available():
                vgg19.cuda()
            # Define the loss functions
            # GAN loss (adversarial loss)
            self.criterionGAN = GANLoss(self.device, gan_mode='vanilla')
            # MSE loss
            self.criterionMSE = torch.nn.MSELoss()
            # Pixel loss (perceptual loss)
            self.criterionPixel = PixelLoss(vgg19, self.device)
            # Define the optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if torch.cuda.is_available():
                vgg19.cuda()
        else:
            web_dir = os.path.join(opt.results_dir, opt.name, str(opt.load_epoch))
            self.image_paths = ['{}/images/{}.png'.format(web_dir, 'reconstructed'),
                            '{}/images/gtruth.png'.format(web_dir),
                            '{}/images/blurred.png'.format(web_dir)]
    # Set the input for the model
    def set_input(self, input):
        """
        Sets the input data for the model.

        Args:
            input (tuple): A tuple containing the input tensors (real_B, real_C).
        """
        if OPERATION_MODE==OperationMode.INFERENCE.value or OPERATION_MODE==OperationMode.TESTING.value or OPERATION_MODE==OperationMode.VALIDATION.value or OPERATION_MODE==OperationMode.EVALUATION.value:
          X_mb = input
          self.real_B = X_mb.to(self.device)
        else:
          X_mb, y_mb = input[0], input[1]
          self.real_B = X_mb.to(self.device)
          self.real_C = y_mb.to(self.device)
        
    # Forward pass
    def forward(self):
        """
        Performs a forward pass through the generator to produce fake images.
        """
        self.fake_C = self.netG(self.real_B)    

    def compute_visuals(self):
        self.fake_C = self.fake_C[0,0,:,:]
        self.real_C = self.real_C[0,0,:,:]
        self.real_B = self.real_B[0,0:self.depth//2,:,:].squeeze(0)
       
    # backward pass for the discriminator
    def backward_D(self):
        """
        Calculates the loss for the discriminator and performs a backward pass to update the discriminator's weights.
        """
        # set requires_grad to True for Discriminator
        self.set_requires_grad(self.netD, True)
        # set requires_grad to False for Generator
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()
        gen_C = self.netG(self.real_B)
        pred_fake = self.netD(gen_C)
        # Calculate the loss for the fake images
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD(self.real_C)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()
        self.optimizer_D.step()
        for p in self.netD.parameters():
            p.data.clamp_(-0.01, 0.01)

    # backward pass for the generator
    def backward_G(self):
        """
        Calculates the loss for the generator and performs a backward pass to update the generator's weights.
        """
        # set requires_grad to False for Discriminator
        self.set_requires_grad(self.netD, False)
        # set requires_grad to True for Generator
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        gen_C = self.netG(self.real_B)
        pred_fake = self.netD(gen_C)
        # Calculate the loss for the fake images
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.ladv
        self.loss_G_MSE = self.criterionMSE(gen_C, self.real_C) * self.opt.lmse
        self.loss_G_Perc = self.criterionPixel(gen_C, self.real_C) * self.opt.lperc
        self.loss_G = self.loss_G_GAN + self.loss_G_MSE + self.loss_G_Perc
        self.loss_G.backward()
        self.optimizer_G.step()

    # save the model
    def save_models(self):
        """
        Saves the current state of the generator and discriminator models.
        """
        torch.save(self.netG.state_dict(), os.path.join(OUTPUT_DIR, 'netG.pth'))
        if self.isTrain:
          torch.save(self.netD.state_dict(), os.path.join(OUTPUT_DIR, 'netD.pth'))
    
    # load the model
    def load_models(self):
        """
        Loads the state of the generator and discriminator models.
        """
        self.netG.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'netG.pth')))
        if self.isTrain:
          self.netD.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'netD.pth')))

    def save_every_batch(self):
        """
        Saves the current state of the generator and discriminator models after each batch.
        """
        torch.save(self.netG.state_dict(), os.path.join(BATCH_OUTPUT_DIR, 'batch_netG.pth'))
        torch.save(self.netD.state_dict(), os.path.join(BATCH_OUTPUT_DIR, 'batch_netD.pth'))
        
    def load_every_batch(self):
        """
        Loads the state of the generator and discriminator models after each batch.
        """

        self.netG.load_state_dict(torch.load(os.path.join(BATCH_OUTPUT_DIR, 'batch_netG.pth')))
        self.netD.load_state_dict(torch.load(os.path.join(BATCH_OUTPUT_DIR, 'batch_netD.pth')))

    # optimize the parameters    
    def optimize_parameters(self):
        """
        Optimizes the parameters of the generator and discriminator models.
        """
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    

if __name__ == "__main__":
    
    opt=TrainOptions()
    model = DenoiserGan(opt=opt)
    x = torch.randn((1,1,512,389))
    y = torch.randn((1,1,512,389))
    model.set_input((x, y))
    model.forward()
    faks=model.fake_C
    print(faks.shape)
    print(faks.max(), faks.min())
 
    

# python -m src.denoiser.models.gan_model
    