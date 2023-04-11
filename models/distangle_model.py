import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util

class DistangleModel(BaseModel):
    def name(self):
        return 'DistangleModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(netG='RESNEXT')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G']
        self.model_names = ['G']
        
        opt.output_nc= 3 if opt.task=='sr' else 1 # out channel is 3 for shadow removal, out channel is 1 for detection
        self.netG = networks.define_G(4, opt.output_nc, opt.ngf, 'RESNEXT', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG.to(self.device)
        self.netG.print_networks() 
        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.bce = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=1e-5)
            self.optimizers = [self.optimizer_G] #self.optimizers.append()
   
    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_param = input['param'].to(self.device).type(torch.float)
        self.shadowfree_img = input['C'].to(self.device)
            
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.nim = self.input_img.shape[1]
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
    
    def forward(self):
        # compute output of generator
        inputG = torch.cat([self.input_img,self.shadow_mask], 1)
        self.Gout = self.netG(inputG)
        
        # compute lit image
        add = self.Gout[:,[0,2,4]]
        mul = self.Gout[:,[1,3,5]]
        n = self.Gout.shape[0]
        
        add = add.view(n,3,1,1).expand((n,3,256,256))
        mul = mul.view(n,3,1,1).expand((n,3,256,256))
        self.lit = self.input_img.clone()/2+0.5        
        self.lit = self.lit*mul + add

        # compute lit image for ground truth
        addgt = self.shadow_param[:,[0,2,4]]
        mulgt = self.shadow_param[:,[1,3,5]]
        
        addgt = addgt.view(n,3,1,1).expand((n,3,256,256))
        mulgt = mulgt.view(n,3,1,1).expand((n,3,256,256))
        self.litgt = self.input_img.clone()/2+0.5 
        self.litgt = self.litgt*mulgt+addgt
        
        # compute free-shadow image
        self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
        self.out = self.out*2-1

        self.outgt = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.litgt*self.shadow_mask_3d
        self.outgt = self.outgt*2-1
        
        self.alpha = torch.mean(self.shadowfree_img / self.lit,dim=1,keepdim=True)

    def get_prediction(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadow_mask_3d = (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)
        
        # compute output of generator
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        self.Gout = self.netG(inputG)
        
        # compute lit image
        self.lit = self.input_img.clone()/2+0.5
        add = self.Gout[:,[0,2,4]]
        mul = self.Gout[:,[1,3,5]]
        n = self.Gout.shape[0]
        
        add = add.view(n,3,1,1).expand((n,3,256,256))
        mul = mul.view(n,3,1,1).expand((n,3,256,256))
        self.lit = self.lit*mul + add
        
        # compute the free-shadow image (shadow matte = shadow_mask_3d)
        self.out = (self.input_img/2+0.5)*(1-self.shadow_mask_3d) + self.lit*self.shadow_mask_3d
        self.out = self.out*2-1
        return util.tensor2im(self.out,scale =0) 

    def backward_G(self):
        criterion = self.criterionL1 if self.opt.task =='sr' else self.bce
        lambda_ = self.opt.lambda_L1 if self.opt.task =='sr' else 1
        self.loss_G = criterion(self.Gout, self.shadow_param) * lambda_
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     opt = parser.parse_args()
#     opt.dataroot = '=/nfs/bigbox/hieule/GAN/datasets/ISTD_Dataset/train/train_'
#     opt.name = 'test'
#     opt.model = 'jointdistangle'

#     opt.gpu_ids=[2]
#     opt.log_scale = 0
#     opt.ndf = 32
#     opt.ngf = 64
#     opt.norm ='batch'
#     opt.checkpoints_dir ='/nfs/bigbox/hieule/GAN/data/test'  
#     opt.isTrain = False
#     opt.resize_or_crop = 'none'
#     opt.loadSize = 256
#     opt.init_type = 'xavier'
#     opt.init_gain = 0.02
#     opt.fineSize = 256
#     opt.nThreads = 1   # test code only supports nThreads = 1
#     opt.batchSize = 1  # test code only supports batchSize = 1
#     opt.serial_batches = False  # no shuffle
#     opt.no_flip = True  # no flip
#     opt.no_dropout = True
#     opt.use_our_mask = True
#     opt.task ='sr'

#     a = DistangleModel()
#     a.initialize(opt)
