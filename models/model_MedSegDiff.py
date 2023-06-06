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
        #128 size of images for 8GB RAM
        return parser
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_L1', 'G2_param', 'G2_L1']
        self.model_names = ['G1', 'G2']
        
        self.netG1 = define_MedSegDiffNet(opt, 3, 1, timesteps = 4)
        self.netG2 = define_SID(opt, net_g = opt.netS[opt.net2_id[0]], net_m = opt.netG[opt.net2_id[1]])
        
        self.netG1_module = self.netG1.module if len(opt.gpu_ids) > 0 else self.netG1
        self.netG2_module = self.netG2.module if len(opt.gpu_ids) > 0 else self.netG2

        if self.isTrain:
            # Initialize optimizers
            self.MSELoss = torch.nn.MSELoss()
            self.bce = torch.nn.BCEWithLogitsLoss().to(0) # Evaluation value 0 by BCEWithLogitsLoss
            self.criterionL1 = torch.nn.L1Loss().to(1) # Evaluation value 1 by L1Loss'
            
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
        self.fake_shadowmask, self.fake_target = self.netG1(self.input_img, self.shadow_mask) # generally, self.fake_target = self.shadow_mask
        self.fake_shadowmask = (self.fake_shadowmask>0).type(torch.float)*2-1
        
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.fake_shadowmask)
 
    def backward1(self):
        #self.loss_G1_L1 = self.MSELoss(self.fake_shadowmask, self.fake_target)
        #self.loss_G1_L1.backward() 
        self.netG1_module.backward()
        
    def backward2(self):
        lambda_ = 2
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = self.criterionL1 (self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = self.criterionL1 (self.fake_free_shadow_image, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1
        #self.loss_G = self.loss_G1 + self.loss_G2
        self.loss_G2.backward() 

    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.fake_shadowmask = self.netG1_module.get_prediction(self.input_img)
        #self.fake_shadowmask = self.shadow_mask
        self.fake_shadowmask = (self.fake_shadowmask>0).type(torch.float)*2-1
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.fake_shadowmask)
 
        RES = dict()
        RES['final']  = self.fake_free_shadow_image #util.tensor2im(self.final,scale =0)
        RES['phase1'] = self.fake_shadowmask #util.tensor2im(self.out,scale =0)
        #RES['param'] = self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)
        return  RES
    
    def optimize_parameters(self):
        # self.forward()
 
        # self.optimizer_G.zero_grad()
        # self.backward()
        # self.optimizer_G.step()
        
        self.forward()
        
        #self.set_requires_grad([self.netG2_module.netG, self.netG2_module.netM], False)  # Enable backprop for D1, D2
        self.optimizer_G1.zero_grad()
        self.backward1()
        self.optimizer_G1.step()
        
        #self.set_requires_grad([self.netG2_module.netG, self.netG2_module.netM], True)  # Enable backprop for D1, D2
        self.optimizer_G2.zero_grad()
        self.backward2()
        self.optimizer_G2.step()