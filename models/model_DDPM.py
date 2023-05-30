###############################################################################
# This file contains the model class for the combination of STGAN and SID
# STGAN is in charge of detecting shadow. SID is in charge of removing shadow.
###############################################################################

import torch
#import numpy as np
#from torch.autograd import Variable
from .base_model import BaseModel
from .network.network_DDPM import define_DDPMNet

class DDPMModel(BaseModel):
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
        self.loss_names = ['G1_L1']
        self.model_names = ['G1']
        
        self.netG1 = define_DDPMNet(opt, 3, 1)
            
        if self.isTrain:
            # Define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
        
            # Initialize optimizers
            self.optimizers = torch.optim.Adam(self.netG1.module.net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.handmask = (input['handmask'].to(self.device) > 0).type(torch.float)*2-1
        self.handimg = input['handimg'].to(self.device) #Range [-1, 1]
    
    def forward(self, timestep = 0):
        # Compute output of generator 1
        #t = torch.randint(0, self.netG1.module.T, (self.opt.batch_size,), device=self.netG1.module.device).long()
        self.noise_pred, self.noise = self.netG1.module.forward(self.input_img, timestep)

    def backward(self):                                    
        self.loss_G1_L1 = self.criterionL1(self.noise_pred, self.noise)
        self.loss_G =  self.loss_G1_L1
        self.loss_G.backward()
    
    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.pred_img = self.netG1.module.get_prediction()
        #self.forward()

        RES = dict()
        RES['final']  = self.pred_img #util.tensor2im(self.final,scale =0)
        #RES['phase1'] = self.noise #util.tensor2im(self.out,scale =0)
        #RES['param'] = self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)
        return  RES
    
    def optimize_parameters(self):
        self.forward()
        
        self.optimizers.zero_grad()
        self.backward()
        self.optimizers.step()

