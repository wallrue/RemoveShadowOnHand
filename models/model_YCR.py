###############################################################################
# This file contains the model class for the combination of STGAN and SID
# STGAN is in charge of detecting shadow. SID is in charge of removing shadow.
# Example of netG (for STGAN): unet_32, unet_128, unet_256, mobile_unet
# Example of netD: basic, n_layers, pixel
###############################################################################

import torch
from .base_model import BaseModel
from .network import network_GAN
from .network import network_STGAN

class YCRModel(BaseModel):
    def name(self):
        return 'YCrNModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = self.opt.isTrain
        self.loss_names = ['G2_GAN', 'G2_L1', 'D2_real', 'D2_fake']
        self.model_names = ['YCR']
        
        self.netYCR = network_STGAN.define_STGAN(opt, 3+2, 3, net_g = opt.netG[opt.net2_id[0]], net_d = opt.netD[opt.net2_id[1]])
        self.netYCR = self.netYCR.module if len(opt.gpu_ids) > 0 else self.netYCR
        
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.GAN_loss = network_GAN.GANLoss(opt.gpu_ids)
        
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netYCR.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D  = torch.optim.Adam(self.netYCR.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G, self.optimizer_D]
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device).type(torch.float)
        self.input_img_Y = input['shadowfull_Y'].to(self.device).type(torch.float)
        self.input_img_Cr = input['shadowfull_Cr'].to(self.device).type(torch.float)
        self.shadowfree_img = input['shadowfree'].to(self.device).type(torch.float)
        
    def forward(self):
        self.inputnetYCR = torch.cat((self.input_img, self.input_img_Y, self.input_img_Cr), 1)
        self.fake_free_shadow_image = self.netYCR.forward_G(self.inputnetYCR)

    def forward_D(self):
        fake_AB = torch.cat((self.inputnetYCR, self.fake_free_shadow_image), 1)
        real_AB = torch.cat((self.inputnetYCR, self.shadowfree_img), 1)                                                            
        self.pred_fake, self.pred_real = self.netYCR.forward_D(fake_AB.detach(), real_AB)
        
    def backward_D(self):        
        self.loss_D_fake = self.GAN_loss(self.pred_fake, target_is_real = 0) 
        self.loss_D_real = self.GAN_loss(self.pred_real, target_is_real = 1)
        
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()
                                                            
    def backward_G(self):
        self.loss_G_GAN = self.GAN_loss(self.pred_fake, target_is_real = 1)       
        self.loss_G_L1 = self.criterionL1(self.fake_free_shadow_image, self.shadowfree_img)
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*0.1
        self.loss_G.backward()

    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.forward()

        RES = dict()
        RES['final']= self.fake_free_shadow_image
        RES['phase1'] = self.input_img_Cr 
        return  RES
    
    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad(self.netYCR.netD, True)  # Enable backprop for D1, D2
        self.forward_D()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

     
        self.set_requires_grad(self.netYCR.netD, False) # Freeze D
        self.forward_D()   

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

 

        
