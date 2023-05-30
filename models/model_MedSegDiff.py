# -*- coding: utf-8 -*-
"""
Created on Tue May 30 01:38:04 2023

@author: lemin
"""
import torch
from .base_model import BaseModel
from .network.network_MedSegDiff import define_MedSegDiffNet

class MedSegDiffModel(BaseModel):
    def name(self):
        return 'Stacked Conditional Generative Adversarial Networks'
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        return parser
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_L1', 'G2_L1']
        self.model_names = ['G1', 'G2']
        # self.loss_names = ['G2_L1']
        # self.model_names = ['G2']
        
        self.netG1 = define_MedSegDiffNet(opt, 3, 1, timesteps = 3)
        self.netG2 = define_MedSegDiffNet(opt, 4, 3, timesteps = 3)
        
        self.netG1_module = self.netG1.module if len(opt.gpu_ids) > 0 else self.netG1
        self.netG2_module = self.netG2.module if len(opt.gpu_ids) > 0 else self.netG2

        if self.isTrain:
            # Initialize optimizers
            self.MSELoss = torch.nn.MSELoss()
            self.bce = torch.nn.BCEWithLogitsLoss().to(0) # Evaluation value 0 by BCEWithLogitsLoss
            self.criterionL1 = torch.nn.L1Loss().to(1) # Evaluation value 1 by L1Loss'
            self.optimizer_G = torch.optim.Adam([{'params': self.netG1_module.parameters()}, 
                                                  {'params': self.netG2_module.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            # self.optimizer_G = torch.optim.Adam(self.netG2_module.parameters(),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G]
        
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadow_mask = (input['shadowmask'].to(self.device) > 0).type(torch.float)*2-1
        self.shadowfree_img = input['shadowfree'].to(self.device)
        
    def forward(self):
        inputG1 = self.input_img
        self.fake_shadowmask = self.netG1_module.forward(inputG1, self.shadow_mask) 
        #self.fake_shadowmask = self.shadow_mask
        # Compute output of generator 2
        inputG2 = torch.cat((self.input_img, self.fake_shadowmask), 1)
        self.fake_image = self.netG2_module.forward(inputG2, self.shadowfree_img)   
        
    def backward(self):
        self.loss_G1_L1 = self.MSELoss(self.fake_shadowmask, self.shadow_mask)
        self.loss_G2_L1 = self.MSELoss(self.fake_image, self.shadowfree_img)
        self.loss_G = self.loss_G2_L1 + self.loss_G1_L1
        self.loss_G.backward() 

    def get_prediction(self, input_img):
        inputG1 = self.input_img
        self.fake_shadowmask = self.netG1_module.get_prediction(inputG1)
        #self.fake_shadowmask = self.shadow_mask
        inputG2 = torch.cat((self.input_img, self.fake_shadowmask), 1)
        self.fake_img = self.netG2_module.get_prediction(inputG2)     # pass in your unsegmented images
        #self.pred.shape                              # predicted segmented images - (8, 3, 128, 128)
        #self.forward()

        RES = dict()
        RES['final']  = self.fake_img #util.tensor2im(self.final,scale =0)
        RES['phase1'] = self.fake_shadowmask #util.tensor2im(self.out,scale =0)
        #RES['param'] = self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)
        return  RES
    
    def optimize_parameters(self):
        self.forward()
 
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()