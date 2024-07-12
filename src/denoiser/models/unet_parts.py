import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the UNet architecture

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Apply two convolutional layers with ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3 ,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the double convolution layers.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """
        return self.double_conv(x)
# Down-sampling with maxpooling
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
    Down-sampling with maxpooling followed by double convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass through the down-sampling layers.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H/2, W/2).
        """
        return self.maxpool_conv(x)
# Up-sampling with upsampling and double convolution
class Up(nn.Module):
    """
    Up-sampling with bilinear interpolation followed by double convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x1, x2):
        """
        Forward pass through the up-sampling and double convolution layers.

        Args:
            x1 (torch.Tensor): Input tensor from the previous layer of shape (N, in_channels, H/2, W/2).
            x2 (torch.Tensor): Input tensor from the corresponding down-sampling layer of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
# Output convolution 1x1
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, relu=False):
        """
    Output convolution layer with optional ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        relu (bool): Whether to apply ReLU activation after convolution.
    """
        super(OutConv, self).__init__()
        self.has_relu = relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the output convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """
        if self.has_relu:
            x = self.conv(x)
            return self.relu(x)
        else:
            return self.conv(x)