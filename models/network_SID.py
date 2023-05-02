import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import time
import numpy as np
from util.image_pool import ImagePool
import util.util as util
from PIL import ImageOps,Image
from .base_model import BaseModel
from . import network_GAN

class SIDNet(nn.Module):
    def __init__(self):
        super(SIDNet, self).__init__()
        #self.training = istrain
            
        self.netG2 = network_GAN.define_G(opt.input_nc+1, 6, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netM2 = network_GAN.define_G(6+1, opt.output_nc, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

    def forward(self, x):
        # compute output of generator 2
        self.fake_shadow_image = x
        inputG2 = torch.cat([self.input_img, self.fake_shadow_image], 1)
        self.shadow_param_pred = self.netG2(inputG2)
        
        # m = self.shadow_param_pred.shape[1]
        w = inputG2.shape[2]
        h = inputG2.shape[3]
        n = self.shadow_param_pred.shape[0]
        
        # compute lit image
        # self.shadow_param_pred = torch.mean(self.shadow_param_pred.view([n,m,-1]),dim=2)
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = (self.shadow_param_pred[:,[1,3,5]]*2) +3
        
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        self.lit = self.input_img.clone()/2+0.5
        self.lit = self.lit*mul + add

        # compute shadow matte
        #lit.detach if no final loss for paramnet 
        inputM2 = torch.cat([self.input_img, self.lit, self.fake_shadow_image],1)
        self.alpha_pred = self.netM2(inputM2)
        self.alpha_pred = (self.alpha_pred +1) /2 
        
        # compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = self.final*2-1
        
        return self.shadow_param_pred, self.alpha_pred, self.final
