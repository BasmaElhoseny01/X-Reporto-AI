import torch
from torch import nn
from torch.optim import lr_scheduler

# Define the loss functions (adversarial and pixel-wise loss)
class GANLoss(nn.Module):
    def __init__(self, device, gan_mode='wgangp', target_real_label=1.0, target_fake_label=0.0):
        """
        Define the GAN loss for different GAN modes.

        Args:
            device (torch.device): Device to run the computations on.
            gan_mode (str): Type of GAN loss. Options: 'wgangp', 'lsgan', 'vanilla'.
            target_real_label (float): Label for real images.
            target_fake_label (float): Label for fake images.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.device = device
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode {} not implemented'.format(gan_mode))
    
    def get_target_tensor(self, prediction, target_is_real):
        """
        Get the target tensor for the given prediction.

        Args:
            prediction (torch.Tensor): The prediction from the discriminator.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            torch.Tensor: The target tensor expanded to the size of the prediction.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor.to(self.device))
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class PixelLoss(nn.Module):
    def __init__(self, feature_extractor, device):
        """
        Calculate the GAN loss.

        Args:
            prediction (torch.Tensor): The prediction from the discriminator.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            torch.Tensor: The calculated loss.
        """
        super(PixelLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.device = device
        
    
    def __call__(self, prediction, target):
        """
        Calculate the pixel-wise loss.

        Args:
            prediction (torch.Tensor): The predicted image.
            target (torch.Tensor): The target image.

        Returns:
            torch.Tensor: The calculated loss.
        """
        vggf_gt = self.feature_extractor(torch.cat([target, target, target], 1)).to(self.device)
        vggf_gen = self.feature_extractor(torch.cat([prediction, prediction, prediction], 1)).to(self.device)
        loss = nn.MSELoss()
        return loss(vggf_gt, vggf_gen)

# Define the scheduler
def get_scheduler(optimizer, opt):
    """
    Get the learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        opt: Options including lr_policy (str), epoch_count (int), n_epochs (int),
             n_epochs_decay (int), lr_decay_iters (int).

    Returns:
        lr_scheduler._LRScheduler: The learning rate scheduler.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler