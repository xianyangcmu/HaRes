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
class HaResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activation = 'relu', scale = torch.tensor(0.1, dtype = torch.float64)):
        super(HaResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, padding_mode = 'zeros', padding = 1)
        self.activate = activation_func(activation)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, padding_mode = 'zeros', padding = 1)
        self.shortcut = nn.Identity()
        self.scale = scale
    
    def forward(self, x, ha, mask):
        ha = ha * mask
        residual = self.shortcut(ha)
        #print("residual", residual.shape)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = torch.mul(x, self.scale)
        #print("x", x.shape)
        x += residual
        
        return x
    
class EDSR_TRI_LR(nn.Module):
    def __init__(self, config):
        super(EDSR_TRI_LR, self).__init__()
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
        #self.upsample = nn.Upsample(scale_factor=self.factor, mode='bilinear')
        self.activate = activation_func(activation)
        self.res_l = nn.ModuleList()
        self.shortcut = nn.Identity()
        for i in range(num_layers):
            self.res_l.append(ResBlock(out_channels, out_channels, kernel_size))
        self.conv_pred = nn.Conv2d(out_channels, out_channels, kernel_size, padding_mode = 'zeros', padding = 1)
        # final layers
        self.pred_l = nn.ModuleList()
        for i in range(3):
            #self.pred_l.append(ResBlock(out_channels, out_channels, kernel_size))
            self.pred_l.append(HaResBlock(out_channels, out_channels, kernel_size))
        #self.fc
        
        
    
    def forward(self, landsat, sentinel, cloudy, ground_truth, mask, coeffs = None, intercepts = None, stds = None):
        
        
        rem = self.shortcut(cloudy)

        landsat, cloudy, sentinel = landsat.double(), cloudy.double(), sentinel.double()
        #interpolate

        #landsat = self.upsample(landsat)
        landsat = F.interpolate(landsat,scale_factor=self.factor, mode='nearest')
        
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
        # concat linear regression results as a feature
        r, g, b = landsat[:,0:1,:,:], landsat[:,1:2,:,:], landsat[:,2:3,:,:]
        pred_r, pred_g, pred_b = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        coeff = self.config['std_factor']
        lam = self.config['reg_lambda']
        if self.config['rgb_feature']:
            if not self.config['inplace']:
                # harmonization loss
                blue_b = -0.0107
                blue_a = 1.0946
                blue_std = 0.0128

                green_b = 0.0026
                green_a = 1.0043
                green_std = 0.0136

                red_b = -0.0015
                red_a = 1.0524
                red_std = 0.0163
                #r_, g_, b_ = ground_truth[:,0:1,:,:], ground_truth[:,1:2,:,:], ground_truth[:,2:3,:,:]


                r_mean = red_a * r + red_b
                g_mean = green_a * g + green_b
                b_mean = blue_a * b + blue_b
                
                rgb_feat = torch.cat((r_mean, g_mean, b_mean), 1)
                #x = torch.cat((x, rgb_feat), 1)
                #x = x + rgb_feat
                #x = self.conv_pred(x)
                for i in range(len(self.pred_l)):
                        
                        #x = self.pred_l[i](x)
                    x = self.pred_l[i](x, rgb_feat, mask)
                x = self.conv_pred(x)
            else:
                if self.config['univariate']:
                    r_coeff, g_coeff, b_coeff = coeffs[:,0:1],coeffs[:,1:2],coeffs[:,2:3]
                    r_intercept, g_intercept, b_intercept = intercepts[:,0:1],intercepts[:,1:2],intercepts[:,2:3]
                    r_std, g_std, b_std = stds[:,0:1],stds[:,1:2],stds[:,2:3]


                    r_mean = r * r_coeff + r_intercept
                    g_mean = g * g_coeff + g_intercept
                    b_mean = b * b_coeff + b_intercept
                    rgb_feat = torch.cat((r_mean, g_mean, b_mean), 1)
                    #x = torch.cat((x, rgb_feat), 1)
                    #x = x + rgb_feat
                    #x = self.conv_pred(x)
                    for i in range(len(self.pred_l)):
                        
                        #x = self.pred_l[i](x)
                        x = self.pred_l[i](x, rgb_feat, mask)
                    x = self.conv_pred(x)
                elif not self.config['univariate']:
                    r_coeff, g_coeff, b_coeff = coeffs[:,0],coeffs[:,1],coeffs[:,2]
                    r_intercept, g_intercept, b_intercept = intercepts[:,0:1],intercepts[:,1:2],intercepts[:,2:3]
                    r_std, g_std, b_std = stds[:,0:1],stds[:,1:2],stds[:,2:3]
                    # x: batch, 3, 384, 384
                    # coeff: batch, 3, 384, 384
                    #print(x.shape, r_coeff.shape, r_intercept.shape, (x*r_coeff).shape, torch.sum(x * r_coeff, axis = 1,keepdim = True).shape)
                    r_mean = torch.sum(x * r_coeff, axis = 1, keepdim = True) + r_intercept
                    g_mean = torch.sum(x * g_coeff, axis = 1, keepdim = True) + g_intercept
                    b_mean = torch.sum(x * b_coeff, axis = 1, keepdim = True) + b_intercept
                    rgb_feat = torch.cat((r_mean, g_mean, b_mean), 1)
                    #x = torch.cat((x, rgb_feat), 1)
                    x = x + rgb_feat
                    #x = self.conv_pred(x)
                    for i in range(len(self.pred_l)):
                        #x = self.pred_l[i](x, rgb_feat)
                        x = self.pred_l[i](x)
                
                    
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
        mse = mse_loss(x, ground_truth)
    
        
        if not self.config['inplace']:
            # harmonization loss
            blue_b = -0.0107
            blue_a = 1.0946
            blue_std = 0.0128

            green_b = 0.0026
            green_a = 1.0043
            green_std = 0.0136

            red_b = -0.0015
            red_a = 1.0524
            red_std = 0.0163
            #r_, g_, b_ = ground_truth[:,0:1,:,:], ground_truth[:,1:2,:,:], ground_truth[:,2:3,:,:]
            

            r_mean = red_a * r + red_b
            r_l, r_u = r_mean - coeff * red_std, r_mean + coeff * red_std

            g_mean = green_a * g + green_b
            g_l, g_u = g_mean - coeff * green_std, g_mean + coeff * green_std

            b_mean = blue_a * b + blue_b
            b_l, b_u = b_mean - coeff * blue_std, b_mean + coeff * blue_std


            
        else:
            if self.config["univariate"]:
                r_coeff, g_coeff, b_coeff = coeffs[:,0:1],coeffs[:,1:2],coeffs[:,2:3]
                r_intercept, g_intercept, b_intercept = intercepts[:,0:1],intercepts[:,1:2],intercepts[:,2:3]
                r_std, g_std, b_std = stds[:,0:1],stds[:,1:2],stds[:,2:3]


                r_mean = r * r_coeff + r_intercept
                g_mean = g * g_coeff + g_intercept
                b_mean = b * b_coeff + b_intercept
                r_l, r_u = r_mean - coeff * r_std, r_mean + coeff * r_std
                g_l, g_u = g_mean - coeff * g_std, g_mean + coeff * g_std
                b_l, b_u = b_mean - coeff * b_std, b_mean + coeff * b_std
            else:
                
                r_coeff, g_coeff, b_coeff = coeffs[:,0],coeffs[:,1],coeffs[:,2]
                r_intercept, g_intercept, b_intercept = intercepts[:,0:1],intercepts[:,1:2],intercepts[:,2:3]
                r_std, g_std, b_std = stds[:,0:1],stds[:,1:2],stds[:,2:3]
                # x: batch, 3, 384, 384
                # coeff: batch, 3, 384, 384
                #print(x.shape, r_coeff.shape, r_intercept.shape, (x*r_coeff).shape, torch.sum(x * r_coeff, axis = 1,keepdim = True).shape)
                r_mean = torch.sum(x * r_coeff, axis = 1, keepdim = True) + r_intercept
                g_mean = torch.sum(x * g_coeff, axis = 1, keepdim = True) + g_intercept
                b_mean = torch.sum(x * b_coeff, axis = 1, keepdim = True) + b_intercept
                r_l, r_u = r_mean - coeff * r_std, r_mean + coeff * r_std
                g_l, g_u = g_mean - coeff * g_std, g_mean + coeff * g_std
                b_l, b_u = b_mean - coeff * b_std, b_mean + coeff * b_std
                
        r_loss = torch.maximum(torch.maximum(r_l - pred_r, pred_r - r_u), torch.zeros_like(r))
        g_loss = torch.maximum(torch.maximum(g_l -pred_g, pred_g - g_u), torch.zeros_like(g))
        b_loss = torch.maximum(torch.maximum(b_l -pred_b, pred_b - b_u), torch.zeros_like(b))
        harmonization_loss = torch.mean(torch.cat((r_loss, g_loss, b_loss), axis = 1))
        #harmonization_loss = torch.square(harmonization_loss)
        loss = mse + lam * harmonization_loss
        return x, loss, mse, harmonization_loss

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