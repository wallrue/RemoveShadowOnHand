###############################################################################
# This file contains the model class for the combination of STGAN and SID
# STGAN is in charge of detecting shadow. SID is in charge of removing shadow.
# Example of netG (for STGAN): unet_32, unet_128, unet_256, mobile_unet, 
# Example of netG (for SID): resnet_9blocks, resnet_6blocks, RESNEXT, 
#       mobilenetV1, mobilenetV2, mobilenetV3_large, mobilenetV3_small
# Example of netD: basic, n_layers, pixel
###############################################################################

import torch
from .base_model import BaseModel
from .network import network_GAN
from .network import network_STGAN
from .network.network_SIDPAMI import define_SIDPAMI

class SIDPAMISTGANModel(BaseModel):
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
                           'G2_param', 'G2_L1', 'G2_GAN', 'D2_real', 'D2_fake']
        self.model_names = ['G1', 'G2']
        self.netG1 = network_STGAN.define_STGAN(opt, 3, 1, net_g = opt.netG[opt.net1_id[0]], net_d = opt.netD[opt.net1_id[1]])
        self.netG2 = define_SIDPAMI(opt, net_g = opt.netS[opt.net2_id[0]], net_m = opt.netG[opt.net2_id[1]], net_d = opt.netD[opt.net1_id[1]])
            
        #self.netG1.to(self.device)
        #self.netG2.to(self.device)
        
        self.netG1_module = self.netG1.module if len(opt.gpu_ids) > 0 else self.netG1
        self.netG2_module = self.netG2.module if len(opt.gpu_ids) > 0 else self.netG2
        
        if self.isTrain:
            # Define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            self.GAN_loss = network_GAN.GANLoss(opt.gpu_ids)
        
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'params': self.netG1_module.netG.parameters()}, 
                                                 {'params': self.netG2_module.netG.parameters()},
                                                 {'params': self.netG2_module.netM.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D = torch.optim.Adam([{'params': self.netG1_module.netD.parameters()}, 
                                                 {'params': self.netG2_module.netD.parameters()}],
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
        inputnetG1 = self.input_img
        self.fake_shadow_image = self.netG1_module.forward_G(inputnetG1)
        
        # Compute output of generator 2
        self.fake_shadow_image = (self.fake_shadow_image>0).type(torch.float)*2-1
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2_module.forward_G(self.input_img, self.fake_shadow_image)
                
    def forward_D(self):
        # Compute output of discriminator 1
        fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
        real_AB = torch.cat((self.input_img, self.shadow_mask), 1)                                                            
        self.pred_fake1, self.pred_real1 = self.netG1_module.forward_D(fake_AB.detach(), real_AB)
        
        # Compute output of discriminator 2
        fake_AB = torch.cat((self.input_img, self.fake_free_shadow_image), 1)
        real_AB = torch.cat((self.input_img, self.shadowfree_img), 1)                                                            
        self.pred_fake2, self.pred_real2 = self.netG2_module.forward_D(fake_AB.detach(), real_AB)
        
    def backward1(self):      
        self.loss_D1_fake = self.GAN_loss(self.pred_fake1, target_is_real = 0) 
        self.loss_D1_real = self.GAN_loss(self.pred_real1, target_is_real = 1)
        
        lambda_ = 0.5;
        loss_D1 = self.loss_D1_fake + self.loss_D1_real
        self.loss_D1 = lambda_ * loss_D1
        
        self.loss_D2_fake = self.GAN_loss(self.pred_fake2, target_is_real = 0) 
        self.loss_D2_real = self.GAN_loss(self.pred_real2, target_is_real = 1)
        self.loss_D2 = (self.loss_D2_real + self.loss_D2_fake) * 0.5
        
        self.loss_D = self.loss_D1 + self.loss_D2
        self.loss_D.backward()

    def backward2(self):
        # Calculate gradients for G1----------------------
        lambda1, lambda2  = 2, 10
        self.loss_G1_GAN = self.GAN_loss(self.pred_fake1, target_is_real = 1)                                           
        self.loss_G1_L1 = self.criterionL1(self.fake_shadow_image, self.shadow_mask)
        self.loss_G1 = self.loss_G1_GAN*lambda1 + self.loss_G1_L1*lambda2
        
        # Calculate gradients for G2----------------------
        lambda_ = 2
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = self.criterionL1(self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = self.criterionL1(self.fake_free_shadow_image, self.shadowfree_img) * lambda_
        self.loss_G2_GAN = self.GAN_loss(self.pred_fake2, target_is_real = 1)*lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1 + self.loss_G2_GAN
        
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
        
        self.set_requires_grad([self.netG1_module.netD, self.netG2_module.netD], True)  # Enable backprop for D1, D2
        self.optimizer_D.zero_grad()
        self.forward_D()
        self.backward1()
        self.optimizer_D.step()
     
        self.set_requires_grad([self.netG1_module.netD, self.netG2_module.netD], False) # Freeze D
        self.optimizer_G.zero_grad()
        self.forward_D()
        self.backward2()
        self.optimizer_G.step()

