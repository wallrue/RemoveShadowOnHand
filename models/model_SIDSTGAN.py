###############################################################################
# This file contains the model class for the combination of STGAN and SID
# STGAN is in charge of detecting shadow. SID is in charge of removing shadow.
###############################################################################

import torch
#import numpy as np
#from torch.autograd import Variable
from .base_model import BaseModel
from .network import network_GAN
from .network import network_STGAN
from .network.network_SID import define_SID

class SIDSTGANModel(BaseModel):
    def name(self):
        return 'Shadow Image Decomposition & Stacked Conditional GAN'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_GAN', 'G1_L1', 'D1_real', 'D1_fake', 
                           'G2_param', 'G2_L1']
        self.model_names = ['G1', 'G2']
        
        self.netG1 = network_STGAN.define_STGAN(opt, 3, 1, net_g = 'mobile_unet', net_d = 'n_layers')
        self.netG2 = define_SID(opt, 'mobilenetV3_large', 'unet_256')
            
        #self.netG1.to(self.device)
        #self.netG2.to(self.device)
        
        self.netG1_module = self.netG1.module if len(opt.gpu_ids) > 0 else self.netG1
        
        if self.isTrain:
            # Define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            self.GAN_loss = network_GAN.GANLoss(opt.gpu_ids)
        
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'params': self.netG1_module.netG.parameters()}, 
                                                 {'params': self.netG2.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D = torch.optim.Adam(self.netG1_module.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G, self.optimizer_D]
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadow_mask = input['shadowmask'].to(self.device)
        self.shadow_param = input['shadowparams'].to(self.device).type(torch.float)
        self.shadowfree_img = input['shadowfree'].to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0).type(torch.float)*2-1
        #self.nim = self.input_img.shape[1]
    
    def forward(self):
        # Compute output of generator 1
        inputSTGAN1 = self.input_img
        self.fake_shadow_image = self.netG1_module.forward_G(inputSTGAN1)
        
        # Compute output of generator 2
        self.fake_shadow_image = (self.fake_shadow_image>0).type(torch.float)*2-1
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.fake_shadow_image)
                
    def forward_D(self):
        fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
        real_AB = torch.cat((self.input_img, self.shadow_mask), 1)                                                            
        self.pred_fake, self.pred_real = self.netG1_module.forward_D(fake_AB.detach(), real_AB)
        
    def backward1(self):      
        self.loss_D1_fake = self.GAN_loss(self.pred_fake, target_is_real = 0) 
        self.loss_D1_real = self.GAN_loss(self.pred_real, target_is_real = 1)
        
        lambda_ = 0.5;
        loss_D1 = self.loss_D1_fake + self.loss_D1_real
        self.loss_D = lambda_ * loss_D1
        self.loss_D.backward()

    def backward2(self):
        # Calculate gradients for G1----------------------
        lambda1, lambda2  = 2, 5
        self.loss_G1_GAN = self.GAN_loss(self.pred_fake, target_is_real = 1)                                           
        self.loss_G1_L1 = self.criterionL1(self.fake_shadow_image, self.shadow_mask)
        self.loss_G1 = self.loss_G1_GAN*lambda1 + self.loss_G1_L1*lambda2
        
        # Calculate gradients for G2----------------------
        lambda_ = 5
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = self.criterionL1 (self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = self.criterionL1 (self.fake_free_shadow_image, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1
        
        self.loss_G =  self.loss_G1 + self.loss_G2
        self.loss_G.backward()
    
    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.forward()

        RES = dict()
        RES['final']= self.fake_free_shadow_image #util.tensor2im(self.final,scale =0)
        RES['phase1'] = self.fake_shadow_image #util.tensor2im(self.out,scale =0)
        #RES['param']= self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)
        return  RES
    
    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad(self.netG1_module.netD, True)  # Enable backprop for D1, D2
        self.optimizer_D.zero_grad()
        self.forward_D()
        self.backward1()
        self.optimizer_D.step()
     
        self.set_requires_grad(self.netG1_module.netD, False) # Freeze D
        self.optimizer_G.zero_grad()
        self.forward_D()
        self.backward2()
        self.optimizer_G.step()

