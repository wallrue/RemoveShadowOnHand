import torch
from collections import OrderedDict
import time
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import network_GAN
import util.util as util
from PIL import ImageOps,Image

class STGANModel(BaseModel):
    def name(self):
        return 'Shadow Image Decomposition model ICCV19'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(input_nc=3, output_nc=3)
        parser.set_defaults(checkpoints_dir="C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/checkpoints_STGAN/")
        parser.set_defaults(name='STGAN_PalmHandDataset')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['loss_G1','loss_G2','loss_D1', 'loss_G2']
        self.model_names = ['G1', 'G2', 'D1', 'D2']
        
        self.netG1 = network_GAN.define_G(opt.input_nc, 1, opt.ngf, 'unet_32', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = network_GAN.define_G(4, 3, opt.ngf, 'unet_32', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD1 = network_GAN.define_D(4, opt.ngf, 'n_layers', n_layers_D=3, opt.norm,
                                          use_sigmoid=True, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD2 = network_GAN.define_D(7, opt.ngf, 'n_layers', n_layers_D=3, opt.norm,
                                          use_sigmoid=True, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.netG1.to(self.device)        
        self.netG2.to(self.device)        
        self.netD1.to(self.device)        
        self.netD2.to(self.device)
        
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.bce = nn.BCEWithLogitsLoss().to(0) #Evaluation value 0 by BCEWithLogitsLoss
            self.criterionL1 = nn.L1Loss().to(1) #Evaluation value 1 by L1Loss'
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'params': self.netG1.parameters()}, {'params': self.netG2.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D = torch.optim.Adam([{'params': self.netD1.parameters()}, {'params': self.netD2.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G, self.optimizer_D]
   
    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadowfree_img = input['C'].to(self.device)
        
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
    
    def forward1(self):
        # compute output of generator 1
        inputG1 = self.input_img
        self.fake_shadow_image = self.netG1(inputG1)

        # m = self.shadow_param_pred.shape[1]
        w = inputG1.shape[2]
        h = inputG1.shape[3]
        
        # compute output of generator 2
        inputG2 = torch.cat((self.input_img, self.fake_shadow_image), 1)
        self.fake_free_shadow_image = self.netG2(inputG2)

        
    def backward1(self):      
        """Calculate GAN loss for the discriminator"""
        # update D1, D2, set_requires_grad() = Freeze or not.

        # calculate gradients for D1-----------------
        fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)  # we use GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD1(fake_AB.detach())
   
        ## Real
        real_AB = torch.cat((self.input_img, self.shadow_mask), 1)
        pred_real = self.netD1(real_AB)
        
        # calculate gradients for D2-----------------
        fake_ABC = torch.cat((self.input_img, self.fake_shadow_image, self.fake_free_shadow_image), 1)
        pred_fake2 = self.netD2(fake_ABC.detach())

        ## Real
        real_ABC = torch.cat((self.input_img, self.shadow_mask, self.shadowfree_img), 1)
        pred_real2 = self.netD2(real_ABC)
        
        label_d_fake = np.zeros(pred_fake.size()) #Variable(cuda_tensor(np.zeros(pred_fake.size())), requires_grad=False)
        loss_D1_fake = self.bce(pred_fake, label_d_fake) #input, target
        
        label_d_real = np.ones(pred_fake.size()) #Variable(cuda_tensor(np.ones(pred_fake.size())), requires_grad=False)
        loss_D1_real = self.bce(pred_real, label_d_real)
        
        label_d_fake = np.zeros(pred_fake2.size()) #Variable(cuda_tensor(np.zeros(pred_fake2.size())), requires_grad=False)
        loss_D2_fake = self.bce(pred_fake2, label_d_fake)
        
        label_d_real = np.ones(pred_fake2.size()) #Variable(cuda_tensor(np.ones(pred_real2.size())), requires_grad=False)
        loss_D2_real = self.bce(pred_real2, label_d_real)
        
        lambda1 = 5; lambda2 = 0.1; lambda3 = 0.1;
        self.loss_D1 = loss_D1_fake + loss_D1_real
        self.loss_D2 = loss_D2_fake + loss_D2_real
        self.loss_D = lambda2 * loss_D1 + lambda3 * loss_D2
        self.loss_D.backward()

    def backward2(self):
        # calculate gradients for G1----------------------
        fake_AB = torch.cat((self.input_img, self.fake_shadow_image), 1)
        pred_fake = self.netD1(fake_AB.detach())
        label_d_real = np.ones(pred_fake.size()) #Variable(cuda_tensor(np.ones(pred_fake.size())), requires_grad=False)

        # calculate graidents for G2
        fake_ABC = torch.cat((self.input_img, self.fake_shadow_image, self.fake_free_shadow_image), 1)
        pred_fake = self.netD2(fake_ABC.detach())
        label_d_real = np.ones(pred_fake.size()) #Variable(cuda_tensor(np.ones(pred_fake.size())), requires_grad=False)
        
        loss_G1_GAN = self.bce(pred_fake, label_d_real) 
        loss_G1_L1 = self.criterionL1(self.fake_shadow_image, self.shadow_mask)
        
        loss_G2_GAN = self.bce(pred_fake, label_d_real)
        loss_G2_L1 = self.criterionL1(self.fake_free_shadow_image, self.shadowfree_img)
        
        lambda1 = 5; lambda2 = 0.1; lambda3 = 0.1;
        self.loss_G1 = loss_G1_GAN + loss_G1_L1 * lambda2
        self.loss_G2 = loss_G2_GAN * lambda3 + loss_G2_L1 * lambda1  
        self.loss_G = self.loss_G1 + self.loss_G2
        self.loss_G.backward()  

    def get_prediction(self, input_img, shadow_mask):
        self.input_img = input_img.to(self.device)
        self.shadow_mask = shadow_mask.to(self.device)
        
        self.forward1(self)
        # inputG1 = self.input_img        
        # self.fake_shadow_image = self.netG1(inputG1)
        # inputG2 = torch.cat((self.input_img, self.fake_shadow_image), 1)
        # self.fake_free_shadow_image = self.netG2(inputG2)

        RES = dict()
        RES['final']= self.fake_free_shadow_image #util.tensor2im(self.final,scale =0)
        #RES['phase1'] = util.tensor2im(self.out,scale =0)
        #RES['param']= self.shadow_param_pred.detach().cpu() 
        #RES['matte'] = util.tensor2im(self.alpha_pred.detach().cpu()/2,scale =0)
        return  RES
    
    def optimize_parameters(self):
        self.forward1()
        
        set_requires_grad([D1, D2], True)  # enable backprop for D1, D2
        optimizerD.zero_grad() # set D1's gradients to zero
        self.backward1()
        optimizerD.step()
     
        set_requires_grad([D1, D2], False) #Freeze D
        optimizerG.zero_grad()
        self.backward2()
        optimizerG.step()

