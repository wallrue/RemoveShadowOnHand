# -*- coding: utf-8 -*-
"""
Created on Tue May 30 01:38:04 2023

@author: lemin
"""
import torch
from .base_model import BaseModel
from .network.network_MedSegDiff import define_MedSegDiffNet
from .network.network_SID import define_SID

class MedSegDiffModel(BaseModel):
    def name(self):
        return 'Stacked Conditional Generative Adversarial Networks'
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        #parser.set_defaults(fineSize=64)
        # 128 size of images for 8GB RAM
        return parser
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_L1', 'G2_param', 'G2_L1']
        self.model_names = ['G1', 'G2']
        
        self.netG1 = define_MedSegDiffNet(opt, 3, 1, timesteps = 100)
        self.netG2 = define_SID(opt, net_g = opt.netS[opt.net2_id[0]], net_m = opt.netG[opt.net2_id[1]])
        
        self.netG1_module = self.netG1.module if len(opt.gpu_ids) > 0 else self.netG1
        self.netG2_module = self.netG2.module if len(opt.gpu_ids) > 0 else self.netG2

        if self.isTrain:
            # Initialize optimizers
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            
            self.optimizer_G1 = torch.optim.Adam(self.netG1_module.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G2 = torch.optim.Adam([{'params': self.netG2_module.netG.parameters()},
                                                  {'params': self.netG2_module.netM.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G1, self.optimizer_G2]
        
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadow_mask = (input['shadowmask'].to(self.device) > 0).type(torch.float)*2-1
        self.shadow_param = input['shadowparams'].to(self.device).type(torch.float)
        self.shadowfree_img = input['shadowfree'].to(self.device)
        
    def forward(self):
        self.fake_shadowmask, self.fake_target = self.netG1(self.input_img, self.shadow_mask)
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.fake_shadowmask)
 
    def backward_G1(self):
        self.loss_G1_L1 = self.MSELoss(self.fake_shadowmask, self.fake_target)
        self.loss_G1_L1.backward(retain_graph=False)
        
    def backward_G2(self):
        lambda_ = 2
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = self.criterionL1 (self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = self.criterionL1 (self.fake_free_shadow_image, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1
        self.loss_G2.backward(retain_graph=True)

    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.fake_shadowmask = self.netG1_module.get_prediction(self.input_img)
        self.fake_shadowmask = (self.fake_shadowmask>0).type(torch.float)*2-1
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.fake_shadowmask)
 
        RES = dict()
        RES['final']  = self.fake_free_shadow_image
        RES['phase1'] = self.fake_shadowmask
        return  RES
    
    def optimize_parameters(self):
        self.forward()
        
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()
        
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()