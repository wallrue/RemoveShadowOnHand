###############################################################################
# This file contains definitions of SID Net - Shadow Image Decomposition
# SIDNet is only to remove the shadow but detect shadow mask
###############################################################################

import torch
from torch import nn
from .network_GAN import define_G

class SIDPAMIwINet(nn.Module):
    def __init__(self, opt):
        """ SIDPAMIwINet includes G net, M net and I net, which is to relit and remove shadow 
        from available shadow mask and full shadow image
        """
        super(SIDPAMIwINet, self).__init__()
        #self.training = istrain    
        self.netG = define_G(opt.input_nc+1, 6, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, [])
        self.netM = define_G(6+1, opt.output_nc, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, [])
        self.netI = networks.define_G(6+1, 3, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, [])

    def forward(self, input_img, fake_shadow_image):
        self.input_img = input_img
        self.fake_shadow_image = fake_shadow_image
        inputG = torch.cat([self.input_img, self.fake_shadow_image], 1)
        
        # Compute output of generator 2
        self.shadow_param_pred = self.netG(inputG)
        
        w = inputG.shape[2]
        h = inputG.shape[3]
        n = self.shadow_param_pred.shape[0]
        
        # Compute lit image
        # self.shadow_param_pred = torch.mean(self.shadow_param_pred.view([n,m,-1]),dim=2)
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = (self.shadow_param_pred[:,[1,3,5]]*2) +3
        
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        self.lit = self.input_img.clone()/2+0.5
        self.lit = self.lit*mul + add

        # Compute shadow matte
        inputM = torch.cat([self.input_img, self.lit, self.fake_shadow_image],1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred +1) /2 
        
        # Compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = self.final*2-1
        
        inputI = torch.cat([self.input_img, self.final.detach(), self.fake_shadow_image], 1)
        self.residual = self.netI(inputI)
        self.final_I = self.final+self.residual
        
        return self.shadow_param_pred, self.alpha_pred, self.final, final_I

def define_SIDPAMIwINet(opt):
    net = None
    net = SIDPAMIwINet(opt)
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net