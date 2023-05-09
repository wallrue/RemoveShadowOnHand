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
    def __init__(self, opt):
        super(SIDNet, self).__init__()
        #self.training = istrain    
        self.netG = network_GAN.define_G(opt.input_nc+1, 6, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, [])
        self.netM = network_GAN.define_G(6+1, opt.output_nc, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, [])

    def forward(self, input_img, fake_shadow_image):
        self.input_img = input_img
        self.fake_shadow_image = fake_shadow_image
        inputG = torch.cat([self.input_img, self.fake_shadow_image], 1)
        # compute output of generator 2
        self.shadow_param_pred = self.netG(inputG)
        
        # m = self.shadow_param_pred.shape[1]
        w = inputG.shape[2]
        h = inputG.shape[3]
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
        inputM = torch.cat([self.input_img, self.lit, self.fake_shadow_image],1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred +1) /2 
        
        # compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = self.final*2-1
        
        return self.shadow_param_pred, self.alpha_pred, self.final

def define_SID(opt):
    net = None
    net = SIDNet(opt)
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net