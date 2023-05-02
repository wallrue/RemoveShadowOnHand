import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
import util.util as util
from PIL import ImageOps,Image
from .network_DSD import DSDNet

class DSDSIDModel(BaseModel):
    def name(self):
        return 'Shadow Image Decomposition model ICCV19'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        #parser.set_defaults(checkpoints_dir="C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/checkpoints_DSD/")
        #parser.set_defaults(name='DSD_PalmHandDataset')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_L1', 'G1DST1_L1', 'G1DST2_L1', 'G2_param', 'G2_L1', 'M2']
        self.model_names = ['G1', 'G2']
        self.cuda_tensor = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor
        
        self.netG1 = DSDNet()
        self.netG2 = network_GAN.define_G(opt.input_nc+1, 6, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netM2 = network_GAN.define_G(6+1, opt.output_nc, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG1.to(self.device)
        self.netG2.to(self.device)
        self.netM2.to(self.device)
        
        if self.isTrain:
            #self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            #self.MSELoss = torch.nn.MSELoss()
            #self.bce = torch.nn.BCEWithLogitsLoss().to(0) #Evaluation value 0 by BCEWithLogitsLoss
            self.bce_logit = bce_logit_pred()
            self.bce_logit_dst = bce_logit_dst()
            self.criterionL1 = torch.nn.L1Loss().to(1) #Evaluation value 1 by L1Loss'
            
            # initialize optimizers
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G2 = torch.optim.Adam([{'params': self.netG2.parameters()}, {'params': self.netM2.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G1, self.optimizer_G2]
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadow_mask = input['shadowmask'].to(self.device)
        self.shadowfree_img = input['shadowfree'].to(self.device)
        self.shaded_hand = input['handshaded'].to(self.device)
        self.shadedless_hand = input['handshadedless'].to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
    
    def forward1(self):
        # compute output of generator 1
        inputG1 = self.input_img
        w = inputG1.shape[2]
        h = inputG1.shape[3]
        
        self.fuse_pred_shad, self.pred_down1_shad, self.pred_down2_shad, self.pred_down3_shad, self.pred_down4_shad, \
        self.fuse_pred_dst1, self.pred_down1_dst1, self.pred_down2_dst1, self.pred_down3_dst1, self.pred_down4_dst1, \
        self.fuse_pred_dst2, self.pred_down1_dst2, self.pred_down2_dst2, self.pred_down3_dst2, self.pred_down4_dst2, \
        self.pred_down0_dst1, self.pred_down0_dst2, self.pred_down0_shad = self.netG1(inputG1)
        
        # compute output of generator 2
        self.fake_shadow_image = self.fuse_pred_shad
        self.fake_shadow_image = (self.fake_shadow_image>0.9).type(torch.float)*2-1
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

        # compute shadow matte
        #lit.detach if no final loss for paramnet 
        inputM2 = torch.cat([self.input_img, self.lit, self.fake_shadow_image],1)
        self.alpha_pred = self.netM2(inputM2)
        self.alpha_pred = (self.alpha_pred +1) /2 
        
        # compute free-shadow image
        self.final = (self.input_img/2+0.5)*(1-self.alpha_pred) + self.lit*(self.alpha_pred)
        self.final = self.final*2-1

    def backward1(self):      
        """Calculate GAN loss for the discriminator"""
        # update D1, D2, set_requires_grad() = Freeze or not.
        loss_fuse_shad = self.bce_logit(self.fuse_pred_shad, self.shadowfree_img, self.shaded_hand, self.shadedless_hand)
        loss1_shad = self.bce_logit(self.pred_down1_shad, self.shadowfree_img, self.shaded_hand, self.shadedless_hand)
        loss2_shad = self.bce_logit(self.pred_down2_shad, self.shadowfree_img, self.shaded_hand, self.shadedless_hand)
        loss3_shad = self.bce_logit(self.pred_down3_shad, self.shadowfree_img, self.shaded_hand, self.shadedless_hand)
        loss4_shad = self.bce_logit(self.pred_down4_shad, self.shadowfree_img, self.shaded_hand, self.shadedless_hand)
        loss0_shad = self.bce_logit(self.pred_down0_shad, self.shadowfree_img, self.shaded_hand, self.shadedless_hand)

        loss_fuse_dst1 = self.bce_logit_dst(self.fuse_pred_dst1, self.shaded_hand)
        loss1_dst1 = self.bce_logit_dst(self.pred_down1_dst1, self.shaded_hand)
        loss2_dst1 = self.bce_logit_dst(self.pred_down2_dst1, self.shaded_hand)
        loss3_dst1 = self.bce_logit_dst(self.pred_down3_dst1, self.shaded_hand)
        loss4_dst1 = self.bce_logit_dst(self.pred_down4_dst1, self.shaded_hand)
        loss0_dst1 = self.bce_logit_dst(self.pred_down0_dst1, self.shaded_hand)
        
        loss_fuse_dst2 = self.bce_logit_dst(self.fuse_pred_dst2, self.shadedless_hand)
        loss1_dst2 = self.bce_logit_dst(self.pred_down1_dst2, self.shadedless_hand)
        loss2_dst2 = self.bce_logit_dst(self.pred_down2_dst2, self.shadedless_hand)
        loss3_dst2 = self.bce_logit_dst(self.pred_down3_dst2, self.shadedless_hand)
        loss4_dst2 = self.bce_logit_dst(self.pred_down4_dst2, self.shadedless_hand)
        loss0_dst2 = self.bce_logit_dst(self.pred_down0_dst2, self.shadedless_hand)

        self.loss_G1_L1 = loss_fuse_shad + loss1_shad + loss2_shad + loss3_shad + loss4_shad + loss0_shad
        self.loss_G1DST1_L1 = loss_fuse_dst1 + loss1_dst1 + loss2_dst1 + loss3_dst1 + loss4_dst1 + loss0_dst1
        self.loss_G1DST2_L1 = loss_fuse_dst2 + loss1_dst2 + loss2_dst2 + loss3_dst2 + loss4_dst2 + loss0_dst2
        self.loss_G1 = self.loss_G1_L1 + 2*self.loss_G1DST1_L1 + 2*self.loss_G1DST2_L1
        self.loss_G1.backward()

    def backward2(self):
        criterion = self.criterionL1 
        lambda_ = self.opt.lambda_L1
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = criterion(self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = criterion(self.final, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1
        self.loss_G2.backward()
        
    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        
        self.forward1()
        # inputG1 = self.input_img        
        # self.fake_shadow_image = self.netG1(inputG1)
        # inputG2 = torch.cat((self.input_img, self.fake_shadow_image), 1)
        # self.fake_free_shadow_image = self.netG2(inputG2)

        RES = dict()
        RES['final']= self.fake_free_shadow_image #util.tensor2im(self.final,scale =0)
        RES['phase1'] = self.fake_shadow_image #util.tensor2im(self.out,scale =0)
        #RES['param']= self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)
        return  RES
    
    def optimize_parameters(self):
        self.forward1()
        
        self.set_requires_grad(self.netG2, False)  # enable backprop for D1, D2
        self.set_requires_grad(self.netG1, True)  # enable backprop for D1, D2
        self.optimizer_G1.zero_grad() # set D1's gradients to zero
        self.backward1()
        self.optimizer_G1.step()
     
        self.set_requires_grad(self.netG1, False) #Freeze D
        self.set_requires_grad(self.netG2, True)  # enable backprop for D1, D2
        self.optimizer_G2.zero_grad()
        self.backward2()
        self.optimizer_G2.step()

