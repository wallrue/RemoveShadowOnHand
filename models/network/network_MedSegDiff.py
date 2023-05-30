# -*- coding: utf-8 -*-
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
from einops import rearrange #, reduce

from .network_DiffUNET import Unet, default, identity

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
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
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class MedSegDiffNet(nn.Module):
    def __init__(self, opt, gan_input_nc, gan_output_nc, timesteps = 1000):
        super(MedSegDiffNet, self).__init__()
        self.image_size = opt.fineSize
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if len(opt.gpu_ids)>0 else torch.device('cpu')

        sampling_timesteps = None
        ddim_sampling_eta = 1.
        self.netG = Unet(   dim = 64,
                            image_size = self.image_size,
                            mask_channels = gan_output_nc,          # segmentation has 1 channel
                            input_img_channels = gan_input_nc,     # input images have 3 channels
                            dim_mults = (1, 2, 4, 8)) #model if isinstance(model, Unet) else model.module

        self.input_img_channels = self.netG.input_img_channels
        self.mask_channels = self.netG.mask_channels
        self.self_condition = self.netG.self_condition

        #--- Define targets ---------------------------
        self.objective = 'pred_noise'
        assert self.objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        #--- Create Noise ---------------------------
        beta_schedule = 'cosine'
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda val: val.to(torch.float32)
        self.betas = register_buffer(betas)
        self.alphas_cumprod = register_buffer(alphas_cumprod)
        self.alphas_cumprod_prev = register_buffer(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = register_buffer(torch.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = register_buffer(torch.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = register_buffer(torch.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = register_buffer(torch.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = register_buffer(torch.sqrt(1. / alphas_cumprod - 1))
        
        # register_buffer('betas', betas)
        # register_buffer('alphas_cumprod', alphas_cumprod)
        # register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # # calculations for diffusion q(x_t | x_{t-1}) and others
        # register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.terior_variance = register_buffer(self.posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.posterior_log_variance_clipped = register_buffer(torch.log(self.posterior_variance.clamp(min =1e-20)))
        self.posterior_mean_coef1 = register_buffer(betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = register_buffer((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        # register_buffer('posterior_variance', posterior_variance)

        # # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        # register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        # register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
            
        # if self.isTrain:
        #     # Initialize optimizers
        #     self.MSELoss = torch.nn.MSELoss()
        #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
        #                                         lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
        #     self.optimizers = [self.optimizer_G]
            
    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, c, x_self_cond = None, clip_x_start = False):
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

    def p_mean_variance(self, x, t, c, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, c, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, c = c, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, cond_img, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, cond_img):
        batch_size, device = cond_img.shape[0], self.device
        cond_img = cond_img.to(self.device)

        image_size, mask_channels = self.image_size, self.mask_channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, image_size, image_size), cond_img)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, input_img, output_img):
        self.input_img, self.output_img = input_img, output_img
        if self.input_img.ndim == 3:
            self.input_img = rearrange(self.input_img, 'b h w -> b 1 h w')

        if self.output_img.ndim == 3:
            self.output_img = rearrange(self.output_img, 'b h w -> b 1 h w')

        b, c, h, w, device, img_size, img_channels, mask_channels = *self.input_img.shape, self.device, self.image_size, self.input_img_channels, self.mask_channels

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        assert self.input_img.shape[1] == img_channels, f'your input medical must have {img_channels} channels'
        assert self.output_img.shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels'

        self.times = torch.randint(0, self.num_timesteps, (b,), device = self.device).long()

        self.output_img = normalize_to_neg_one_to_one(self.output_img)
        
        #---------------p_loss
        x_start, t, cond, noise = self.output_img, self.times, self.input_img, None
        
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

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
            
        return self.fake_target
    
    # def backward(self):
    #     self.loss_G2_L1 = self.MSELoss(self.fake_target, self.target)
    #     self.loss_G2_L1.backward()
    @torch.no_grad()    
    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.pred_img = self.sample(self.input_img)     # pass in your unsegmented images
        #self.pred.shape                              # predicted segmented images - (8, 3, 128, 128)
        #self.forward()
        return self.pred_img
    
def define_MedSegDiffNet(opt, gan_input_nc, gan_output_nc, timesteps = 1000):
    net = MedSegDiffNet(opt, gan_input_nc, gan_output_nc, timesteps)
    
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net