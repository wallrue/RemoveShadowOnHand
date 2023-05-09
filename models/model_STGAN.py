import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network_GAN
from . import network_STGAN
import util.util as util
from PIL import ImageOps,Image

class STGANModel(BaseModel):
    def name(self):
        return 'Shadow Image Decomposition model ICCV19'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        #parser.set_defaults(checkpoints_dir="C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/checkpoints_STGAN/")
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G1_GAN', 'G1_L1', 'G2_GAN', 'G2_L1',
                           'D1_real', 'D1_fake', 'D2_real', 'D2_fake']
        self.model_names = ['STGAN1', 'STGAN2']
        self.cuda_tensor = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor
        
        self.netSTGAN1 = network_STGAN.define_STGAN(opt, 3, 1)
        self.netSTGAN2 = network_STGAN.define_STGAN(opt, 4, 3)
        
#         self.STGAN1.to(self.device)        
#         self.STGAN2.to(self.device)
        
        if self.isTrain:
            #self.fake_AB_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.bce = torch.nn.BCEWithLogitsLoss().to(0) #Evaluation value 0 by BCEWithLogitsLoss
            self.criterionL1 = torch.nn.L1Loss().to(1) #Evaluation value 1 by L1Loss'
            self.GAN_loss = network_GAN.GANLoss(opt.gpu_ids)
        
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'params': self.netSTGAN1.module.netG.parameters()}, 
                                                 {'params': self.netSTGAN2.module.netG.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D = torch.optim.Adam([{'params': self.netSTGAN1.module.netD.parameters()}, 
                                                 {'params': self.netSTGAN2.module.netD.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G, self.optimizer_D]
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.shadow_mask = input['shadowmask'].to(self.device)
        self.shadowfree_img = input['shadowfree'].to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
    
    def forward(self):
        # compute output of generator 1
        inputSTGAN1 = self.input_img
        self.fake_shadow_image = self.netSTGAN1.module.forward_G(inputSTGAN1)
        
        # compute output of generator 2
        inputSTGAN2 = torch.cat((self.input_img, self.fake_shadow_image), 1)
        self.fake_free_shadow_image = self.netSTGAN2.module.forward_G(inputSTGAN2)

    def forward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
        real_AB = torch.cat((self.input_img, self.shadow_mask), 1)                                                            
        self.pred_fake, self.pred_real = self.netSTGAN1.module.forward_D(fake_AB.detach(), real_AB)
                                                            
        fake_ABC = torch.cat((self.input_img, self.fake_shadow_image, self.fake_free_shadow_image), 1)
        real_ABC = torch.cat((self.input_img, self.shadow_mask, self.shadowfree_img), 1)                                                   
        self.pred_fake2, self.pred_real2 = self.netSTGAN2.module.forward_D(fake_ABC.detach(), real_ABC)
                                                            
    def backward1(self):
#         fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
#         real_AB = torch.cat((self.input_img, self.shadow_mask), 1)                                                            
#         self.pred_fake, self.pred_real = self.netSTGAN1.module.forward_D(fake_AB.detach(), real_AB)
                                                            
#         fake_ABC = torch.cat((self.input_img, self.fake_shadow_image, self.fake_free_shadow_image), 1)
#         real_ABC = torch.cat((self.input_img, self.shadow_mask, self.shadowfree_img), 1)                                                   
#         self.pred_fake2, self.pred_real2 = self.netSTGAN2.module.forward_D(fake_ABC.detach(), real_ABC)
        
        self.loss_D1_fake = self.GAN_loss(self.pred_fake, target_is_real = 0) 
        self.loss_D1_real = self.GAN_loss(self.pred_real, target_is_real = 1) 
        self.loss_D2_fake = self.GAN_loss(self.pred_fake2, target_is_real = 0)                                                   
        self.loss_D2_real = self.GAN_loss(self.pred_real2, target_is_real = 1) 
                                                            
#         label_d_fake = Variable(self.cuda_tensor(np.zeros(self.pred_fake.size())), requires_grad=False)
#         self.loss_D1_fake = self.bce(self.pred_fake, label_d_fake) #input, target
        
#         label_d_real = Variable(self.cuda_tensor(np.ones(self.pred_fake.size())), requires_grad=False)
#         self.loss_D1_real = self.bce(self.pred_real, label_d_real)
        
#         label_d_fake = Variable(self.cuda_tensor(np.zeros(self.pred_fake2.size())), requires_grad=False)
#         self.loss_D2_fake = self.bce(self.pred_fake2, label_d_fake)
        
#         label_d_real = Variable(self.cuda_tensor(np.ones(self.pred_real2.size())), requires_grad=False)
#         self.loss_D2_real = self.bce(self.pred_real2, label_d_real)
        
        lambda1 = 5; lambda2 = 0.1; lambda3 = 0.1;
        loss_D1 = self.loss_D1_fake + self.loss_D1_real
        loss_D2 = self.loss_D2_fake + self.loss_D2_real
        self.loss_D = lambda2 * loss_D1 + lambda3 * loss_D2
        self.loss_D.backward()

    def backward2(self):
#         fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
#         real_AB = torch.cat((self.input_img, self.shadow_mask), 1)                                                            
#         self.pred_fake, self.pred_real = self.netSTGAN1.module.forward_D(fake_AB.detach(), real_AB)
                                                            
#         fake_ABC = torch.cat((self.input_img, self.fake_shadow_image, self.fake_free_shadow_image), 1)
#         real_ABC = torch.cat((self.input_img, self.shadow_mask, self.shadowfree_img), 1)                                                   
#         self.pred_fake2, self.pred_real2 = self.netSTGAN2.module.forward_D(fake_ABC.detach(), real_ABC)
        
        self.loss_G1_GAN = self.GAN_loss(self.pred_fake, target_is_real = 1)
        self.loss_G2_GAN = self.GAN_loss(self.pred_fake2, target_is_real = 1) 
                                                            
#         fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
#         pred_fake = self.netD1(fake_AB.detach())
#        label_d_real = Variable(self.cuda_tensor(np.ones(self.pred_fake.size())), requires_grad=False)
        
#        self.loss_G1_GAN = self.bce(self.pred_fake, label_d_real) 
        

#         # calculate graidents for G2
#         fake_ABC = torch.cat((self.input_img, self.fake_shadow_image, self.fake_free_shadow_image), 1)
#         pred_fake = self.netD2(fake_ABC.detach())
#        label_d_real = Variable(self.cuda_tensor(np.ones(self.pred_fake.size())), requires_grad=False)
        
#         self.loss_G2_GAN = self.bce(self.pred_fake, label_d_real)
                                                            
        self.loss_G1_L1 = self.criterionL1(self.fake_shadow_image, self.shadow_mask)
        self.loss_G2_L1 = self.criterionL1(self.fake_free_shadow_image, self.shadowfree_img)
        
        lambda1 = 5; lambda2 = 0.1; lambda3 = 0.1;
        loss_G1 = self.loss_G1_GAN + self.loss_G1_L1 * lambda2
        loss_G2 = self.loss_G2_GAN * lambda3 + self.loss_G2_L1 * lambda1  
        self.loss_G = loss_G1 + loss_G2
        self.loss_G.backward()  

    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        
        self.forward()
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
        self.forward()
        
        self.set_requires_grad([self.netSTGAN1.module.netD, self.netSTGAN2.module.netD], True)  # enable backprop for D1, D2
        self.optimizer_D.zero_grad() # set D1's gradients to zero
        self.forward_D()
        self.backward1()
        self.optimizer_D.step()
     
        self.set_requires_grad([self.netSTGAN1.module.netD, self.netSTGAN2.module.netD], False) #Freeze D
        self.optimizer_G.zero_grad()
        self.forward_D()
        self.backward2()
        self.optimizer_G.step()

