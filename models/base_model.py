###############################################################################
# This file contains the base model class which will be inherited by 
# other child model classes
###############################################################################

import os
import torch
from torch.optim import lr_scheduler
import util.util as util
from collections import OrderedDict

class BaseModel():
    def name(self):
        return 'BaseModel'
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
                
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if len(opt.gpu_ids)>0 else torch.device('cpu')
        
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        # When model doesn't vary, we set torch.backends.cudnn.benchmark to get the benefit 
        if opt.resize_or_crop != 'scale_width': 
            torch.backends.cudnn.benchmark = True
            
        self.loss_names = []
        self.model_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadowfree_img = input['C'].to(self.device)
        
        self.nim = self.input_img.shape[1]
        self.shadow_mask_3d= (self.shadow_mask>0).type(torch.float).expand(self.input_img.shape)   
    
    def forward(self):
        pass
    
    def get_prediction(self, input):
        self.input_img = input['A'].to(self.device)
        self.shadow_mask = input['B'].to(self.device)
        self.shadow_mask = (self.shadow_mask>0.9).type(torch.float)*2-1
        self.shadow_mask_3d = (self.shadow_mask>0).type(torch.float).expand(self.inputmg.shape)   
        
        inputG = torch.cat([self.input_img,self.shadow_mask],1)
        out = self.netG(inputG)
        return util.tensor2im(out)

    # Load and print networks; create schedulers
    def setup(self, opt, parser=None):
        print(self.name)
        if self.isTrain:
            print(self.optimizers)
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
        if not self.isTrain:
            print("LOADING %s"%(self.name))
            self.load_networks(opt.epoch)
        self.print_networks()
        
    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'shadow_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70000,90000,13200], gamma=0.3)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    def optimize_parameters(self):
        pass

    # Save and load the networks
    def save_networks(self, epoch):
        for model_name in self.model_names:    
            save_filename = '%s_net_%s.pth' % (epoch, model_name)
            save_path = os.path.join(self.save_dir, save_filename)

            net = getattr(self, 'net' + model_name)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available(): # The case for multiple GPUs
                torch.save(net.module.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # At the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module,key), keys, i + 1) 

    def load_networks(self, epoch):
        for name in self.model_names:
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)

            # The case for multiple GPUs
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
              
            # Loading state dict
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            
            # Loop all keys to find and remove checkpoints of InstanceNorm
            for key in list(state_dict.keys()):
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)
                
    # Print network information
    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        
    def update_learning_rate(self,loss=None):
        for scheduler in self.schedulers:
            if not loss:
                scheduler.step()
            else:
                scheduler.step(loss)
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr
        
    # Set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                     
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if hasattr(self,'loss_'+name):
                errors_ret[name] = float("%.4f" % getattr(self, 'loss_' + name))        
        return errors_ret