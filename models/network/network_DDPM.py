###############################################################################
# This file contains Denoising Diffusion Probabilistic Models (DDPM)
# Fundamentally, Diffusion Models work by destroying training data through 
# the successive addition of Gaussian noise, and then learning to recover 
# the data by reversing this noising process
###############################################################################

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms 
from torch import nn
from .network_UNETDenosing import UnetDenosing


def linear_scheduler(timesteps, start=0.0001, end=0.02):
    """Returns linear schedule for beta
    """
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ Returns values from vals for corresponding timesteps
    while considering the batch dimension.
    
    """
    batch_size = t.shape[0]
    output = vals.gather(-1, t.cpu())
    return output.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, T, device="cpu"):
    """Takes an image and a timestep as input and 
    returns the noisy version of it after adding noise t times.
    """
    betas = linear_scheduler(timesteps=T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# # Define beta schedule
# T = 300
# betas = linear_scheduler(timesteps=T)

# # Pre-calculate different terms for closed form
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, axis=0)
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)



def sample_timestep(x, t, model, T): #@torch.no_grad()
    """Calls the model to predict the noise in the image and returns 
    the denoised image. 
    """
    betas = linear_scheduler(timesteps=T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

def show_tensor_image(image):
    '''Plots image after applying reverse transformations.
    '''
    
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    
def sample_get_image(in_channel, T, model, device, plot = False): #@torch.no_grad()
    # Sample noise
    img_size = 224
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model, T)
        
        if i % stepsize == 0 and plot == True:
            plt.subplot(1, num_images, i/stepsize+1)
            show_tensor_image(img.detach().cpu())
        
    if plot == True: 
        plt.savefig('DDPM_result.png') 
    else:
        return img #Get image when t=0 (real image)
        

class DDPMNet(nn.Module):
    """ DDPM is used for diffusion models for generating high-quality images
    """
    def __init__(self, opt, gan_input_nc, gan_output_nc):
        super(DDPMNet, self).__init__()
        self.opt = opt
        self.input_nc = gan_input_nc
        self.T = 500
        self.device = torch.device(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 else torch.device('cpu')
        self.netG = UnetDenosing(gan_input_nc, gan_output_nc)
        
    def forward(self, input_img, timestep):
        x_noisy, noise = self.forward_diffusion_sample(input_img, timestep, self.T, self.opt.gpu_ids[0])
        noise_pred = self.netG(x_noisy, timestep)
        return noise_pred, noise
    
    def get_loss(self, x_0, t):
        x_noisy, noise = forward_diffusion_sample(x_0, t, self.T, self.opt.gpu_ids[0])
        noise_pred = self.netG(x_noisy, t)
        return F.l1_loss(noise, noise_pred)
    
    def get_prediction(self):
        with torch.no_grad():
            pred_img = sample_get_image(self.input_nc, self.T, self.net, self.opt.gpu_ids[0], plot = False) 
        return pred_img
    
    def plot_samples(self):
        """ Get a noise in random and generate image by the noise
        """
        with torch.no_grad():
            sample_get_image(self.input_nc, self.T, self.net, self.opt.gpu_ids[0], plot = True) #sample_plot_image(IMG_SIZE = 64)
        
def define_DDPMNet(opt, gan_input_nc, gan_output_nc):
    net = DDPMNet(gan_input_nc, gan_output_nc)
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net

