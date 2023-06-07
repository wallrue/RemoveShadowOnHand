import torch
from .base_model import BaseModel
from .network import network_GAN
from .network import network_STGAN

class STGANwHandModel(BaseModel):
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
        self.cuda_tensor = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor
        
        self.netSTGAN1 = network_STGAN.define_STGAN(opt, 3, 1, net_g = opt.netG[opt.net1_id[0]], net_d = opt.netD[opt.net1_id[1]])
        self.netSTGAN2 = network_STGAN.define_STGAN(opt, 4, 3, net_g = opt.netG[opt.net2_id[0]], net_d = opt.netD[opt.net2_id[1]])
        
        self.netSTGAN1 = self.netSTGAN1.module if len(opt.gpu_ids) > 0 else self.netSTGAN1
        self.netSTGAN2 = self.netSTGAN2.module if len(opt.gpu_ids) > 0 else self.netSTGAN2
        
        self.loss_names = ['G1_GAN', 'G1_L1', 'G2_GAN', 'G2_L1',
                           'D1_real', 'D1_fake', 'D2_real', 'D2_fake']
        self.model_names = ['STGAN1', 'STGAN2']
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss().to(1)
            self.GAN_loss = network_GAN.GANLoss(opt.gpu_ids)
        
            # Initialize optimizers
            self.optimizer_G1 = torch.optim.Adam(self.netSTGAN1.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_G2 = torch.optim.Adam(self.netSTGAN2.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizer_D  = torch.optim.Adam([{'params': self.netSTGAN1.netD.parameters()},
                                                  {'params': self.netSTGAN2.netD.parameters()}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
            self.optimizers = [self.optimizer_G1, self.optimizer_G2, self.optimizer_D]
   
    def set_input(self, input):
        self.input_img = input['shadowfull'].to(self.device)
        self.hand_mask = (input['handmask'].to(self.device) > 0).type(torch.float)*2-1
        
        self.hand_img = input['handimg'].to(self.device) #Range [-1, 1]
        self.hand_img = input['shadowfull'].to(self.device) * (self.hand_mask < 0) + self.hand_img
    
    def forward(self):
        # Compute output of generator 1
        inputSTGAN1 = self.input_img
        self.fake_hand_mask = self.netSTGAN1.forward_G(inputSTGAN1)
        
        # Compute output of generator 2
        inputSTGAN2 = torch.cat((self.input_img, self.fake_hand_mask), 1)
        self.fake_hand_img = self.netSTGAN2.forward_G(inputSTGAN2)

    def forward_D(self):
        fake_AB = torch.cat((self.input_img, self.fake_hand_mask), 1)
        real_AB = torch.cat((self.input_img, self.hand_mask), 1)                                                            
        self.pred_fake, self.pred_real = self.netSTGAN1.forward_D(fake_AB.detach(), real_AB)
                                                            
        fake_ABC = torch.cat((self.input_img, self.fake_hand_mask, self.fake_hand_img), 1)
        real_ABC = torch.cat((self.input_img, self.hand_mask, self.hand_img), 1)                                                   
        self.pred_fake2, self.pred_real2 = self.netSTGAN2.forward_D(fake_ABC.detach(), real_ABC)
                                                            
    def backward_D(self):        
        self.loss_D1_fake = self.GAN_loss(self.pred_fake, target_is_real = 0) 
        self.loss_D1_real = self.GAN_loss(self.pred_real, target_is_real = 1) 
        self.loss_D2_fake = self.GAN_loss(self.pred_fake2, target_is_real = 0) 
        self.loss_D2_real = self.GAN_loss(self.pred_real2, target_is_real = 1) 
        
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real)*0.1
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real)*0.1
        self.loss_D = self.loss_D1 + self.loss_D2*5
        self.loss_D.backward()
        
    def backward_G1(self):
        self.loss_G1_GAN = self.GAN_loss(self.pred_fake, target_is_real = 1)                                                
        self.loss_G1_L1 = self.criterionL1(self.fake_hand_mask, self.hand_mask)
        self.loss_G1 = self.loss_G1_GAN + self.loss_G1_L1*0.1
        self.loss_G1.backward(retain_graph=False)

    def backward_G2(self):
        self.loss_G2_GAN = self.GAN_loss(self.pred_fake2, target_is_real = 1)    
        self.loss_G2_L1 = self.criterionL1(self.fake_hand_img, self.hand_img)
        
        self.loss_G2 = self.loss_G2_GAN + self.loss_G2_L1*0.1
        self.loss_G2.backward(retain_graph=True)

    def get_prediction(self, input_img):
        self.input_img = input_img.to(self.device)
        self.forward()

        RES = dict()
        RES['final'] = self.fake_hand_img
        RES['phase1'] = self.fake_hand_mask 
        return  RES
    
    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad([self.netSTGAN1.netD, self.netSTGAN2.netD], True)  # Enable backprop for D1, D2
        self.forward_D()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad([self.netSTGAN1.netD, self.netSTGAN2.netD], False) # Freeze D
        self.forward_D()   
        
        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()
             
        self.optimizer_G1.zero_grad()
        self.backward_G1()
        self.optimizer_G1.step()

