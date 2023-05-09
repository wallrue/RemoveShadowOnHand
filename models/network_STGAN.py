import torch
import torch.nn.functional as F
from torch import nn
import time
import numpy as np
from torch.autograd import Variable
from util.image_pool import ImagePool
from . import network_GAN
import util.util as util
from PIL import ImageOps,Image

class STGANNet(nn.Module):
    def __init__(self, opt, gan_input_nc, gan_output_nc):
        super(STGANNet, self).__init__()
        self.netG = network_GAN.define_G(gan_input_nc, gan_output_nc, opt.ngf, 'unet_32', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, [])
        self.netD = network_GAN.define_D(gan_input_nc+gan_output_nc, opt.ngf, 'n_layers', 3, opt.norm, 
                                         True, opt.init_type, opt.init_gain, [])

    def forward_G(self, input_img):
        fake_image = self.netG(input_img)
        return fake_image
    
    def forward_D(self, fake_package, real_package): 
        pred_fake = self.netD(fake_package)
        pred_real = self.netD(real_package)
        
        return pred_fake, pred_real

def define_STGAN(opt, gan_input_nc, gan_output_nc):
    net = None
    net = STGANNet(opt, gan_input_nc, gan_output_nc)
    
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net

