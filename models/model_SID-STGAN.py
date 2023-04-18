import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network_GAN
import util.util as util
from PIL import ImageOps,Image

class SIDModel(BaseModel):
    def name(self):
        return 'Shadow Image Decomposition model ICCV19'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        parser.set_defaults(checkpoints_dir="C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/checkpoints_SID-STGAN/")
        # parser.set_defaults(netG='RESNEXT')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_GAN', 'G1_L1', 'D1_real', 'D1_fake', 'G2_param', 'G2_GAN', 'M2']
        self.model_names = ['G1', 'D1', 'G2', 'M2']
        
        self.netG1 = network_GAN.define_G(opt.input_nc, 1, opt.ngf, 'unet_32', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD1 = network_GAN.define_D(4, opt.ngf, 'n_layers', 3, opt.norm, True, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.netG2 = network_GAN.define_G(opt.input_nc+1, 6, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netM2 = network_GAN.define_G(6+1, opt.output_nc, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG1.to(self.device)
        self.netD1.to(self.device)
        self.netG2.to(self.device)
        self.netM2.to(self.device)
        
        if self.isTrain:
            #self.fake_AB_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'params': self.netG1.parameters()}, {'params': self.netG2.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D = torch.optim.Adam(self.netD1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_M = torch.optim.Adam(self.netM.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G, self.optimizer_M, self.optimizer_D]
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadow_mask = input['shadowmask'].to(self.device)
        self.shadow_param = input['shadowparams'].to(self.device).type(torch.float)
        self.shadowfree_img = input['shadowfree'].to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
        self.shadow_mask_3d = (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)
    
    def forward(self):
        inputG1 = self.input_img
        self.fake_shadow_image = self.netG1(inputG1)
        
        # compute output of generator
        inputG2 = torch.cat([self.input_img, self.fake_shadow_image], 1)
        self.shadow_param_pred = self.netG2(inputG2)

        # m = self.shadow_param_pred.shape[1]
        w = inputG2.shape[2]
        h = inputG2.shape[3]
        n = self.shadow_param_pred.shape[0]
        
        # compute lit image
        # self.shadow_param_pred = torch.mean(self.shadow_param_pred.view([n,m,-1]),dim=2)
        add = self.shadow_param_pred[:,[0,2,4]]
        mul = (self.shadow_param_pred[:,[1,3,5]]*2) +3
        
        add = add.view(n,3,1,1).expand((n,3,w,h))
        mul = mul.view(n,3,1,1).expand((n,3,w,h))
        self.lit = self.input_img.clone()/2+0.5
        self.lit = self.lit*mul + add
        
#         # compute lit image for ground truth
#         addgt = self.shadow_param[:,[0,2,4]]
#         mulgt = self.shadow_param[:,[1,3,5]]

#         addgt = addgt.view(n,3,1,1).expand((n,3,w,h))
#         mulgt = mulgt.view(n,3,1,1).expand((n,3,w,h))
        
#         self.litgt = self.input_img.clone()/2+0.5
#         self.litgt = (self.litgt*mulgt+addgt)*2-1
        
#         # compute relit image
#         self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
#         self.out = self.out*2-1

        # compute shadow matte
        #lit.detach if no final loss for paramnet 
        inputM2 = torch.cat([self.input_img, self.lit, self.shadow_mask],1)
        self.alpha_pred = self.netM2(inputM2)
        self.alpha_pred = (self.alpha_pred +1) /2 
        
        # compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = self.final*2-1
                
    def backward1(self):      
        """Calculate GAN loss for the discriminator"""
        # update D1, D2, set_requires_grad() = Freeze or not.

        # calculate gradients for D1-----------------
        fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)  # we use GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD1(fake_AB.detach())
   
        ## Real
        real_AB = torch.cat((self.input_img, self.shadow_mask), 1)
        pred_real = self.netD1(real_AB)
        
        label_d_fake = Variable(self.cuda_tensor(np.zeros(pred_fake.size())), requires_grad=False)
        self.loss_D1_fake = self.bce(pred_fake, label_d_fake) #input, target
        label_d_real = Variable(self.cuda_tensor(np.ones(pred_fake.size())), requires_grad=False)
        self.loss_D1_real = self.bce(pred_real, label_d_real)
        
        lambda_ = self.opt.lambda_L1
        loss_D1 = self.loss_D1_fake + self.loss_D1_real
        self.loss_D1 = lambda_ * loss_D1        
        self.loss_D1.backward()

    def backward2(self):
        # calculate gradients for G1----------------------
        lambda1, lambda2  = 5, 20
        fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
        pred_fake = self.netD1(fake_AB.detach())
        label_d_real = Variable(self.cuda_tensor(np.ones(pred_fake.size())), requires_grad=False)
        self.loss_G1_GAN = self.bce(pred_fake, label_d_real) 
        self.loss_G1_L1 = self.criterionL1(self.fake_shadow_image, self.shadow_mask)
        self.loss_G1 = self.loss_G1_GAN*lambda1 + self.loss_G1_L1*lambda2
                
        criterion = self.criterionL1 
        lambda_ = self.opt.lambda_L1
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = criterion(self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_GAN = criterion(self.final, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_GAN
        
        self.loss_G =  self.loss_G1 + self.loss_G2
        self.loss_G.backward()
    
    def get_prediction(self, input_img, shadow_mask):
        self.input_img = input_img.to(self.device)
        self.shadow_mask = shadow_mask.to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadow_mask_3d = (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
        
        w = self.input_img.shape[2]
        h = self.input_img.shape[3]
        n = self.input_img.shape[0]
        m = self.input_img.shape[1]
        
        inputG1 = self.input_img
        self.fake_shadow_image = self.netG1(inputG1)
        
        # compute output of generator
        inputG2 = torch.cat([self.input_img, self.fake_shadow_image],1)
        inputG2 = F.interpolate(inputG2,size=(256,256))
        self.shadow_param_pred = self.netG2(inputG2)
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
        inputM2 = torch.cat([self.input_img, self.lit, self.shadow_mask],1)
        self.alpha_pred = self.netM2(inputM2)
        self.alpha_pred = (self.alpha_pred +1) /2        
        #self.alpha_pred_3d=  self.alpha_pred.repeat(1,3,1,1)
 
        # compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*self.alpha_pred
        self.final = self.final*2-1

        RES = dict()
        RES['final']= self.final #util.tensor2im(self.final,scale =0)
        RES['phase1'] = self.fake_shadow_image #util.tensor2im(self.out,scale =0)
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
    
    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad([self.netD1], True)  # enable backprop for D1, D2
        self.optimizer_D.zero_grad() # set D1's gradients to zero
        self.backward1()
        self.optimizer_D.step()
     
        self.set_requires_grad([self.netD1], False) #Freeze D
        self.optimizer_G.zero_grad()
        self.optimizer_M.zero_grad()
        self.backward2()
        self.optimizer_G.step()
        self.optimizer_M.step()

