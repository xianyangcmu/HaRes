import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from skimage.transform import resize
class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation = 'relu', scale = torch.tensor(0.1, dtype = torch.float64)):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, padding_mode = 'zeros', padding = 1)
        self.activate = activation_func(activation)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, padding_mode = 'zeros', padding = 1)
        self.shortcut = nn.Identity()
        self.scale = scale
    
    def forward(self, x):
        residual = self.shortcut(x)
        #print("residual", residual.shape)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = torch.mul(x, self.scale)
        #print("x", x.shape)
        x += residual
        
        return x
    
class FusionResModel_interp(nn.Module):
    def __init__(self, config):
        super(FusionResModel_interp, self).__init__()
        self.config = config
        in_channels, out_channels = self.config['in_channels'], self.config['out_channels']
        feat = self.config['feat']
        activation = self.config['activation']
        kernel_size = self.config['kernel_size']
        num_layers = self.config['num_layers']
        img_channels = self.config['img_channels']
        if self.config['use_mask']:
            self.conv = nn.Conv2d(3 * img_channels + 1, out_channels, kernel_size, padding_mode = 'zeros', padding = 1)
            self.deConv = nn.ConvTranspose2d(img_channels, img_channels, kernel_size, stride=2, padding=1, output_padding = 1)
        else:
            self.conv = nn.Conv2d(3 * img_channels, out_channels, kernel_size, padding_mode = 'zeros', padding = 1)
            self.deConv = nn.ConvTranspose2d(img_channels, img_channels, kernel_size, stride=2, padding=1, output_padding = 1)
        self.factor = self.config['downsample_factor']
        self.upsample = nn.Upsample(scale_factor=self.factor, mode='nearest')
        self.activate = activation_func(activation)
        self.res_l = nn.ModuleList()
        self.shortcut = nn.Identity()
        for i in range(num_layers):
            self.res_l.append(ResBlock(out_channels, out_channels, kernel_size))
        #self.fc
        
        
    
    def forward(self, landsat, sentinel, cloudy, ground_truth, mask):
        #print("x",x.dtype)
        rem = self.shortcut(cloudy)

        landsat, cloudy, sentinel = landsat.double(), cloudy.double(), sentinel.double()
        #interpolate

        landsat = self.upsample(landsat)

        #deconv

        #prev = self.deConv(prev)

        x = torch.cat((landsat, cloudy, sentinel), 1)
        x = x.double()
        # if single image as input
        #x = x[:,8:12,:,:]
        #ground_truth = ground_truth[:,8:12,:,:]
        #mask = mask[:,8:12,:,:]
        
#         rem = self.shortcut(x)
        # concat mask
        mask_c = mask[:,0:1,:,:]
        if self.config['use_mask']:
            x = torch.cat((x, mask_c), 1)
        x = self.conv(x)
        x = self.activate(x)
        for i in range(len(self.res_l)):
            x = self.res_l[i](x)
        x += rem
        
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        ground_truth = ground_truth.double()
        # original loss
#         window = self.config['temporal_window']
#         channels = 4
#         start = window * channels
#         end = start + channels
#         print(start, end)
        loss = mse_loss(x, ground_truth)
        # cloudy loss
        #print("Before loss: ", mask.shape)
        #cloudy_loss = self.my_loss(x, ground_truth, mask, start, end)
        #free_loss = self.my_loss(x, ground_truth, 1. - mask, start, end)
#         print(cloudy_loss, free_loss)
        #loss = cloudy_loss * self.config['cloudy_weight'] + free_loss * self.config['free_weight']

#         print(loss)
#         print("################")
        return x, loss

    def my_loss(self, x, ground_truth, mask, start = 8, end = 12):
        x = x[:,start:end, :, :]
        ground_truth = ground_truth[:,start:end,:,:]
        mask = mask[:,start:end,:,:]
        #print("After loss: ", mask.shape)
        npixel = torch.sum(mask, dim=(1, 2, 3)) / 4
        sample_loss = torch.sum((x - ground_truth) ** 2 * mask, dim=(1, 2, 3))
        loss = torch.mean(sample_loss / npixel)
        return loss

    
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]