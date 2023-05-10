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
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_L1', 'G1DST1_L1', 'G1DST2_L1', 'G2_param', 'G2_L1']
        self.model_names = {'G1', 'G2'}
        self.cuda_tensor = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor
        
        self.netG1 = define_DSD(opt)
        self.netG2 = define_SID(opt)
            
        self.netG1.to(self.device)
        self.netG2.to(self.device)
        
        if self.isTrain:
            self.bce_logit = bce_logit_pred
            self.bce_logit_dst = bce_logit_dst
            self.criterionL1 = torch.nn.L1Loss().to(1) # Evaluation value 1 by L1Loss
            
            # Initialize optimizers
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G1, self.optimizer_G2]
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadowmask_img = input['shadowmask'].to(self.device)
        self.shadowfree_img = input['shadowfree'].to(self.device)
        self.shaded_hand = input['handshaded'].to(self.device)
        self.shadedless_hand = input['handshadedless'].to(self.device)
        self.shadow_param = input['shadowparams'].to(self.device).type(torch.float)
        
        self.shadowmask_img = (self.shadowmask_img>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
    
    def forward1(self):
        # Compute output of generator 1
        inputG1 = self.input_img
        self.fuse_pred_shad, self.pred_down1_shad, self.pred_down2_shad, self.pred_down3_shad, self.pred_down4_shad, \
        self.fuse_pred_dst1, self.pred_down1_dst1, self.pred_down2_dst1, self.pred_down3_dst1, self.pred_down4_dst1, \
        self.fuse_pred_dst2, self.pred_down1_dst2, self.pred_down2_dst2, self.pred_down3_dst2, self.pred_down4_dst2, \
        self.pred_down0_dst1, self.pred_down0_dst2, self.pred_down0_shad = self.netG1(inputG1)
        
        # Compute output of generator 2
        self.fake_shadow_image = self.fuse_pred_shad
        self.fake_shadow_image = (self.fake_shadow_image>0.9).type(torch.float)*2-1
        self.shadow_param_pred, self.alpha_pred, self.fake_free_shadow_image = self.netG2(self.input_img, self.fake_shadow_image)

    def backward1(self):
        loss_fuse_shad = self.bce_logit(self.fuse_pred_shad, self.shadowmask_img, self.shaded_hand, self.shadedless_hand)
        loss1_shad = self.bce_logit(self.pred_down1_shad, self.shadowmask_img, self.shaded_hand, self.shadedless_hand)
        loss2_shad = self.bce_logit(self.pred_down2_shad, self.shadowmask_img, self.shaded_hand, self.shadedless_hand)
        loss3_shad = self.bce_logit(self.pred_down3_shad, self.shadowmask_img, self.shaded_hand, self.shadedless_hand)
        loss4_shad = self.bce_logit(self.pred_down4_shad, self.shadowmask_img, self.shaded_hand, self.shadedless_hand)
        loss0_shad = self.bce_logit(self.pred_down0_shad, self.shadowmask_img, self.shaded_hand, self.shadedless_hand)

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
        lambda_ = 100
        self.shadow_param[:,[1,3,5]] = (self.shadow_param[:,[1,3,5]])/2 - 1.5
        self.loss_G2_param = self.criterionL1 (self.shadow_param_pred, self.shadow_param) * lambda_ 
        self.loss_G2_L1 = self.criterionL1 (self.fake_free_shadow_image, self.shadowfree_img) * lambda_
        self.loss_G2 = self.loss_G2_param + self.loss_G2_L1
        self.loss_G2.backward()
        
    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.forward1()

        RES = dict()
        RES['final']= self.fake_free_shadow_image
        RES['phase1'] = self.fake_shadow_image
        #RES['param']= self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)
        return  RES
    
    def optimize_parameters(self):
        self.forward1()
        
        self.set_requires_grad([self.netG2], False)  # Enable backprop for D1, D2
        self.optimizer_G1.zero_grad()
        self.backward1()
        self.optimizer_G1.step()
     
        self.set_requires_grad([self.netG2], True)  # Enable backprop for D1, D2
        self.optimizer_G2.zero_grad()
        self.backward2()
        self.optimizer_G2.step()

