###############################################################################
# This file contains the model class for the combination of DSD and SID
# DSD is in charge of detecting shadow. SID is in charge of removing shadow.
###############################################################################

import torch
from .base_model import BaseModel
from .network.network_DSD import define_DSD, bce_logit_pred, bce_logit_dst
from .network.network_SID import define_SID

class DSDSIDModel(BaseModel):
    def name(self):
        return 'Distraction-aware Shadow Detection & Shadow Image Decomposition'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        parser.set_defaults(fineSize=256)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        
        self.netG1 = define_DSD(opt)
        self.netG2 = define_SID(opt, net_g = opt.netS[opt.net2_id[0]], net_m = opt.netG[opt.net2_id[1]])
            
        self.netG1 = self.netG1.module if len(opt.gpu_ids) > 0 else self.netG1
        self.netG2 = self.netG2.module if len(opt.gpu_ids) > 0 else self.netG2
        
        self.loss_names = ['G1_L1', 'G1DST1_L1', 'G1DST2_L1', 'G2_param', 'G2_L1']
        self.model_names = {'G1', 'G2'}
        if self.isTrain:
            self.bce_logit = bce_logit_pred
            self.bce_logit_dst = bce_logit_dst
            self.criterionL1 = torch.nn.L1Loss().to(1) # Evaluation value 1 by L1Loss
            
            # Initialize optimizers
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G2 = torch.optim.Adam([{'params': self.netG2.netG.parameters()},
                                                  {'params': self.netG2.netM.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G1, self.optimizer_G2]

    def rgb2gray(self, rgb_img): #input size: (4, 4, 3)
        rgb_img = (rgb_img + 1.0)/2.0 
        gray_img = rgb_img[:,:,0]*0.2989 + rgb_img[:,:,1]*0.5870 + rgb_img[:,:,2]*0.1140
        return torch.unsqueeze((gray_img-0.5)*2.0, 2)

    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadow_param = input['shadowparams'].to(self.device).type(torch.float)
        self.shadowmask_img = input['shadowmask'].to(self.device)
        self.shadowmask_img = (self.shadowmask_img > 0).type(torch.float)*2-1
        
        #this will be extracted at first: hand segmentation
        self.handmask = self.shadowmask_img
        self.shadeless_inhand = ((self.shadowmask_img > 0)*(self.handmask < 0)).type(torch.float)*2-1
        
        #create non-shadow on hand image 
        self.shadowfree_img = input['shadowfree'].to(self.device)

    def forward(self):
        # Compute output of generator 1
        inputG1 = self.input_img
        self.fuse_pred_shad, self.pred_down1_shad, self.pred_down2_shad, self.pred_down3_shad, self.pred_down4_shad, \
        self.fuse_pred_dst1, self.pred_down1_dst1, self.pred_down2_dst1, self.pred_down3_dst1, self.pred_down4_dst1, \
        self.fuse_pred_dst2, self.pred_down1_dst2, self.pred_down2_dst2, self.pred_down3_dst2, self.pred_down4_dst2, \
        self.pred_down0_dst1, self.pred_down0_dst2, self.pred_down0_shad = self.netG1(inputG1)
        
        # Compute output of generator 2
        self.fake_shadow_image = self.fuse_pred_shad
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.fake_shadow_image)

    def backward_G1(self):
        loss_fuse_shad = self.bce_logit(self.fuse_pred_shad, self.shadowmask_img, self.shadeless_inhand, self.handmask)
        loss1_shad = self.bce_logit(self.pred_down1_shad, self.shadowmask_img, self.shadeless_inhand, self.handmask)
        loss2_shad = self.bce_logit(self.pred_down2_shad, self.shadowmask_img, self.shadeless_inhand, self.handmask)
        loss3_shad = self.bce_logit(self.pred_down3_shad, self.shadowmask_img, self.shadeless_inhand, self.handmask)
        loss4_shad = self.bce_logit(self.pred_down4_shad, self.shadowmask_img, self.shadeless_inhand, self.handmask)
        loss0_shad = self.bce_logit(self.pred_down0_shad, self.shadowmask_img, self.shadeless_inhand, self.handmask)

        loss_fuse_dst1 = self.bce_logit_dst(self.fuse_pred_dst1, self.shadeless_inhand)
        loss1_dst1 = self.bce_logit_dst(self.pred_down1_dst1, self.shadeless_inhand)
        loss2_dst1 = self.bce_logit_dst(self.pred_down2_dst1, self.shadeless_inhand)
        loss3_dst1 = self.bce_logit_dst(self.pred_down3_dst1, self.shadeless_inhand)
        loss4_dst1 = self.bce_logit_dst(self.pred_down4_dst1, self.shadeless_inhand)
        loss0_dst1 = self.bce_logit_dst(self.pred_down0_dst1, self.shadeless_inhand)
        
        loss_fuse_dst2 = self.bce_logit_dst(self.fuse_pred_dst2, self.handmask)
        loss1_dst2 = self.bce_logit_dst(self.pred_down1_dst2, self.handmask)
        loss2_dst2 = self.bce_logit_dst(self.pred_down2_dst2, self.handmask)
        loss3_dst2 = self.bce_logit_dst(self.pred_down3_dst2, self.handmask)
        loss4_dst2 = self.bce_logit_dst(self.pred_down4_dst2, self.handmask)
        loss0_dst2 = self.bce_logit_dst(self.pred_down0_dst2, self.handmask)

        self.loss_G1_L1 = loss_fuse_shad + loss1_shad + loss2_shad + loss3_shad + loss4_shad + loss0_shad
        self.loss_G1DST1_L1 = loss_fuse_dst1 + loss1_dst1 + loss2_dst1 + loss3_dst1 + loss4_dst1 + loss0_dst1
        self.loss_G1DST2_L1 = loss_fuse_dst2 + loss1_dst2 + loss2_dst2 + loss3_dst2 + loss4_dst2 + loss0_dst2
        self.loss_G1 = self.loss_G1_L1 + 2*self.loss_G1DST1_L1 + 2*self.loss_G1DST2_L1
        self.loss_G1.backward(retain_graph=False)

    def backward_G2(self):
        lambda_ = 1
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = self.criterionL1 (self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = self.criterionL1 (self.fake_free_shadow_image, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1
        self.loss_G2.backward(retain_graph=True)
        
    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.forward()

        RES = dict()
        RES['final']= self.fake_free_shadow_image
        RES['phase1'] = self.fake_shadow_image
        return  RES
    
    def optimize_parameters(self):
        self.forward()
          
        self.set_requires_grad([self.netG2.netG, self.netG2.netM], True)  # Enable backprop for D1, D2
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()
        
        self.set_requires_grad([self.netG2.netG, self.netG2.netM], False)  # Enable backprop for D1, D2
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()


