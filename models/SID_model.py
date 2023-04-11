import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
from .distangle_model import DistangleModel
from PIL import ImageOps,Image
class SIDModel(DistangleModel):
    def name(self):
        return 'Shadow Image Decomposition model ICCV19'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G_param','alpha','rescontruction']
        self.model_names = ['G','M']
        opt.output_nc= 3 
        self.netG = networks.define_G(4, 6, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netM = networks.define_G(7, 3, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG.to(self.device)
        self.netM.to(self.device)
        
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_M = torch.optim.Adam(self.netM.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_M)
   
    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.shadowfree_img = input['C'].to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
        self.shadow_mask_3d = (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)
    
    def forward(self):
        # compute output of generator
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        #self.shadow_param_pred = torch.squeeze(self.netG(inputG))
        self.shadow_param_pred = self.netG(inputG)

        # m = self.shadow_param_pred.shape[1]
        w = inputG.shape[2]
        h = inputG.shape[3]
        
        # compute lit image
        # self.shadow_param_pred = torch.mean(self.shadow_param_pred.view([n,m,-1]),dim=2)
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = (self.shadow_param_pred[:,[1,3,5]]*2) +3
        n = self.shadow_param_pred.shape[0]
        #mul = (mul +2) * 5/3          
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        self.lit = self.input_img.clone()/2+0.5
        self.lit = self.lit*mul + add
        
        # compute lit image for ground truth
        addgt = self.shadow_param[:,[0,2,4]]
        mulgt = self.shadow_param[:,[1,3,5]]

        addgt = addgt.view(n,3,1,1).expand((n,3,w,h))
        mulgt = mulgt.view(n,3,1,1).expand((n,3,w,h))
        
        self.litgt = self.input_img.clone()/2+0.5
        self.litgt = (self.litgt*mulgt+addgt)*2-1
        
        # compute relit image
        self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
        self.out = self.out*2-1

        # compute shadow matte
        #lit.detach if no final loss for paramnet 
        inputM = torch.cat([self.input_img,self.lit,self.shadow_mask],1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred +1) /2 
        
        # compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = self.final*2-1

    
    def get_prediction(self, input_img, shadow_mask):
        self.input_img = input_img.to(self.device)
        self.shadow_mask = shadow_mask.to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadow_mask_3d = (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        
        w = self.input_img.shape[2]
        h = self.input_img.shape[3]
        n = self.input_img.shape[0]
        m = self.input_img.shape[1]
        
        # compute output of generator
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        inputG = F.interpolate(inputG,size=(256,256))
        self.shadow_param_pred = self.netG(inputG)
        self.shadow_param_pred = self.shadow_param_pred.view([n,6,-1])
        self.shadow_param_pred = torch.mean(self.shadow_param_pred,dim=2)
        self.shadow_param_pred[:,[1,3,5]] = (self.shadow_param_pred[:,[1,3,5]]*2)+3 
 
        # compute lit image
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = self.shadow_param_pred[:,[1,3,5]]
        #mul = (mul +2) * 5/3          
        n = self.shadow_param_pred.shape[0]
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        self.lit = self.input_img.clone()/2+0.5
        self.lit = self.lit*mul + add
        
        # compute relit image
        self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
        self.out = self.out*2-1
        
        # compute shadow matte        
        inputM = torch.cat([self.input_img,self.lit,self.shadow_mask],1)
        self.alpha_pred = self.netM(inputM)
        self.alpha_pred = (self.alpha_pred +1) /2        
        #self.alpha_pred_3d=  self.alpha_pred.repeat(1,3,1,1)
 
        # compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*self.alpha_pred
        self.final = self.final*2-1

        RES = dict()
        RES['final']= util.tensor2im(self.final,scale =0)
        #RES['phase1'] = util.tensor2im(self.out,scale =0)
        #RES['param']= self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)

        '''
        ###EVAL on original size
        input_img_ori = input['A_ori'].to(self.device)
        input_img_ori = input_img_ori/2+0.5
        lit_ori = input_img_ori
        w = input_img_ori.shape[2]
        h = input_img_ori.shape[3]
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = self.shadow_param_pred[:,[1,3,5]]
        #mul = (mul +2) * 5/3          
        n = self.shadow_param_pred.shape[0]
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        lit_ori = lit_ori*mul + add
        alpha_pred = F.upsample(self.alpha_pred,(w,h),mode='bilinear',align_corners=True)
        final  = input_img_ori * (1-alpha_pred) + lit_ori*(alpha_pred)
        final = final*2 -1 
        RES['ori_Size'] = util.tensor2im(final.detach().cpu())
        '''
        return  RES
        
    def backward(self):
        criterion = self.criterionL1 
        lambda_ = self.opt.lambda_L1
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G_param = criterion(self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_rescontruction = criterion(self.final,self.shadowfree_img) * lambda_
        self.loss = self.loss_G_param + self.loss_rescontruction
        self.loss.backward()
    
    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_M.zero_grad()
        self.backward()
        self.optimizer_G.step()
        self.optimizer_M.step()

