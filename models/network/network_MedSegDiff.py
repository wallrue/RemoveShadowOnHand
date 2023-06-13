###############################################################################
# Original code from
# https://github.com/lucidrains/med-seg-diff-pytorch/blob/main/med_seg_diff_pytorch/med_seg_diff_pytorch.py
###############################################################################
"""
Created on Tue May 30 18:34:11 2023

@author: lemin
"""
import math
from random import random
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from tqdm import tqdm
from functools import partial
from einops import rearrange
from .network_GAN import ResUNet

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def identity(t, *args, **kwargs):
    return t

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta schedule
def linear_beta_schedule(timesteps):
    # Default: 1000 steps in range [0.0001 to 0.01]*2 (forware, backward)
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]     #Normalize to [1, .. , 0] - decrease
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  #Normalize to 1 - [a[1]/a[0], a[2]/ a[1], .. a[-1]/a[-2]]
    return torch.clip(betas, 0, 0.999)

class MedSegDiffNet(nn.Module):
    def __init__(self, opt, gan_input_nc, gan_output_nc, timesteps = 1000):
        super(MedSegDiffNet, self).__init__()
        self.image_size = opt.fineSize
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if len(opt.gpu_ids)>0 else torch.device('cpu')

        self.netG = ResUNet(    dim = 64,
                                image_size = self.image_size,
                                mask_channels = gan_output_nc,          # input image channels
                                input_img_channels = gan_input_nc,      # input image channels
                                dim_mults = (1, 2, 4, 8))               # depth size of Unet

        self.input_img_channels = self.netG.input_img_channels
        self.mask_channels = self.netG.mask_channels
        self.self_condition = self.netG.self_condition

        self.objective = 'pred_x0' # define targets: noise (xT), x_start (x0) and v (parameter v)
        assert self.objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        #--- Create Noise ---------------------------
        beta_schedule = 'cosine'
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas # betas in [0, ..., 1] => alphas in [1, ..., 0]
        alphas_cumprod = torch.cumprod(alphas, dim=0) # cumulative product of elements
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.) # one-padding the last dimension by (1,0)

        timesteps, = betas.shape #timesteps is length of point lists
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = timesteps #default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps

        # helper function to register buffer from float64 to float32
        cuda_tensor = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor
        register_buffer = lambda val: val.type(cuda_tensor) #.to(torch.float32).to(self.device)
        self.betas = register_buffer(betas)
        self.alphas_cumprod = register_buffer(alphas_cumprod)
        self.alphas_cumprod_prev = register_buffer(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = register_buffer(torch.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = register_buffer(torch.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = register_buffer(torch.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = register_buffer(torch.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = register_buffer(torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = register_buffer(self.posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = register_buffer(torch.log(self.posterior_variance.clamp(min =1e-20)))
        self.posterior_mean_coef1 = register_buffer(betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = register_buffer((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise): #Predict result images (start) by xt, t and noise
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_v(self, x_t, t, v): #Predict result images (start) by xt, t and v
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def predict_noise_from_start(self, x_t, t, x0): #Predict noise by xt, t and result images (start)
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, c, x_self_cond = None, clip_x_start = False):
        """ Predict the pred_noise and pred_x_start
        """
        model_output = self.netG(x, t, c, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def q_posterior(self, x_start, x_t, t): 
        """ Compute the information of a noised image such as mean, variance, log_variance
        Predict posterior_mean, posterior_variance, posterior_log_variance_clipped 
        from result images (start), x_t and t
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, c, x_self_cond = None, clip_denoised = True):
        """ Predict pred_x_start and compute the information of a pred_x_start
        Predict model_mean, posterior_variance, posterior_log_variance, x_start
        from xt, t, c (original image), x_self_cond (other conditions), clip_denoised
        """
        preds = self.model_predictions(x, t, c, x_self_cond) #result of pre-start prediction
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)
        #Get the parameters of prediction x_start, xt and t
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond = None, clip_denoised = True):
        """ Create prediction image from model_mean and model_log_variance
        """
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, c = c, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        img = torch.randn(shape, device = self.betas.device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond)

        img = (img+1)*0.5 #Normalize to [0, 1]
        return img

    @torch.no_grad()
    def sample(self, cond_img):
        batch_size = cond_img.shape[0]
        cond_img = cond_img #.to(self.device)

        image_size, mask_channels = self.image_size, self.mask_channels
        sample_fn = self.p_sample_loop # if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, image_size, image_size), cond_img)

    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_v(self, x_start, t, noise): #Predict v by noise, t and result images (start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    def forward(self, input_img, output_img):
        self.input_img, self.output_img = input_img, output_img
        if self.input_img.ndim == 3:
            self.input_img = rearrange(self.input_img, 'b h w -> b 1 h w')

        if self.output_img.ndim == 3:
            self.output_img = rearrange(self.output_img, 'b h w -> b 1 h w')

        b, c, h, w = self.input_img.shape

        assert h == self.image_size and w == self.image_size, f'height and width of image must be {self.image_size}'
        assert self.input_img.shape[1] == self.input_img_channels, f'your input medical must have {self.input_img_channels} channels'
        assert self.output_img.shape[1] == self.mask_channels, f'the segmented image must have {self.mask_channels} channels'

        self.times = torch.randint(0, self.num_timesteps, (b,), device = self.device).long() # Create time list
        self.output_img = self.output_img*2-1 # Normalize to [-1, 1]
        
        #---------------p_loss
        x_start, t, cond = self.output_img, self.times, self.input_img
        noise = torch.randn_like(x_start)
        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # predicting x_0
                x_self_cond = self.model_predictions(x, t, cond).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        self.fake_target = self.netG(x, t, cond, x_self_cond)

        if self.objective == 'pred_noise':
            self.target = noise
        elif self.objective == 'pred_x0':
            self.target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            self.target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        return self.fake_target, self.target
    
    @torch.no_grad()    
    def get_prediction(self, input_img):
        self.input_img = input_img
        self.pred_img = self.sample(self.input_img)     # pass in your unsegmented images
        return self.pred_img
    
def define_MedSegDiffNet(opt, gan_input_nc, gan_output_nc, timesteps = 1000):
    net = MedSegDiffNet(opt, gan_input_nc, gan_output_nc, timesteps)
    
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net