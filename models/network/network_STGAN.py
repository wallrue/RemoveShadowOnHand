###############################################################################
# This file contains definitions of Stacked Conditional GAN (STGAN)
###############################################################################

import torch
from torch import nn
from .network_GAN import define_G, define_D

class STGANNet(nn.Module):
    """ STGAN is built from two GANs. This class is the definition of a 
    single GAN architecture, which includes a generator (Unet32) and 
    a discriminator (PatchGAN discriminator)
    """
    def __init__(self, opt, gan_input_nc, gan_output_nc):
        super(STGANNet, self).__init__()
        self.netG = define_G(gan_input_nc, gan_output_nc, opt.ngf, 'unet_32', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, [])
        self.netD = define_D(gan_input_nc+gan_output_nc, opt.ngf, 'n_layers', 3, opt.norm, 
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
