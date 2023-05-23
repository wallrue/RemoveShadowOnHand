###############################################################################
# This file contains Denoising Diffusion Probabilistic Models (DDPM)
# Fundamentally, Diffusion Models work by destroying training data through 
# the successive addition of Gaussian noise, and then learning to recover 
# the data by reversing this noising process
###############################################################################

import torch
from torch import nn
from torch.optim import Adam
from .network_UNETDenosing import UnetDenosing, forward_diffusion_sample


# Define beta schedule
T = 300
betas = linear_scheduler(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

def sample_timestep(x, t): #@torch.no_grad()
    """Calls the model to predict the noise in the image and returns 
    the denoised image. 
    """
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
    
def sample_plot_image(): #@torch.no_grad()
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i/stepsize+1)
            show_tensor_image(img.detach().cpu())
    plt.show()            

class DDPMNet(nn.Module):
    """ DDPM is used for diffusion models for generating high-quality images
    """
    def __init__(self, opt, gan_input_nc):
        super(DDPMNet, self).__init__()
        self.opt = opt
        if gan_input_nc == 3:
            self.net = UnetDenosing()
        else:
            raise NotImplementedError('DDPM should have 3 input channels, 1 output channel')
        
    def forward(self, input_img, timestep):
        x_noisy, noise = forward_diffusion_sample(input_img, timestep, self.opt.gpu_ids[0])
        noise_pred = self.net(x_noisy, timestep)
        return noise_pred, noise
    
    def plot_samples():
        """ Get a noise in random and generate image by the noise
        """
        with torch.no_grad():
            sample_plot_image()
        
def define_DDPMNet(opt, gan_input_nc):
    net = DDPMNet(gan_input_nc)
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net

