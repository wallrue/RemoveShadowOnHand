import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot',  help='path to images')
        parser.add_argument('--num_threads', default=2, type=int, help='# threads for loading data, num_workers')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='channels of input image')
        parser.add_argument('--output_nc', type=int, default=3, help='channels of output image')
        
#         parser.add_argument('--keep_ratio', action='store_true')
#         parser.add_argument('--norm_mean', type=list, default=[0.5,0.5,0.5])
#         parser.add_argument('--norm_std', type=list, default=[0.5,0.5,0.5])
        
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        #parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='single', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, help='chooses which model to use. cycle_gan, pix2pix, test')
        #parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--lambda_GAN', type=float, default=0.0)
        parser.add_argument('--lambda_smooth', type=float, default=0.0)
        parser.add_argument('--lambda_L1', type=float, default=0.0)
        parser.add_argument('--finetuning', action='store_true')
        parser.add_argument('--finetuning_name', type=str)
        parser.add_argument('--finetuning_epoch', type=str)
        parser.add_argument('--finetuning_dir', type=str)

        self.initialized = True
        return parser
    
    def get_known(self, parser):
                
        parser.set_defaults(batchs=16)
        parser.set_defaults(lr=0.0002)
        parser.set_defaults(GPU=0)
        parser.set_defaults(loadSize=256)
        parser.set_defaults(fineSize=256)
        
        parser.set_defaults(model="SID")
        parser.set_defaults(netG='RESNEXT')
        parser.set_defaults(phase='train_')
        parser.set_defaults(gpu_ids='0')
        parser.set_defaults(dataset_mode='shadowparam')
        parser.set_defaults(save_epoch_freq=2)
        
        parser.set_defaults(niter=10)
        parser.set_defaults(niter_decay=40)
        parser.set_defaults(lambda_L1=100)
        parser.set_defaults(name='SID_GRESNEXT_shadowparam')
        
        parser.set_defaults(checkpoints_dir="C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/checkpoints_PAMI/")
        parser.set_defaults(dataroot='C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/data_processing/dataset/NTUST_TU/train/')
        parser.set_defaults(mask_train='C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/data_processing/dataset/NTUST_TU/train/' + 'train_B')
        parser.set_defaults(param_path='C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/data_processing/dataset/NTUST_TU/train/' + 'train_params')
        
        args, unknown = parser.parse_known_args()
        return args

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(self.parserparser)

        # get the basic options
        #opt, _ = parser.parse_known_args()
        opt = self.get_known(parser)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt = self.get_known(parser)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        
        self.parser = parser
        
        args, unknown = self.parser.parse_known_args()

        return args

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
