from .base_options import BaseOptions


class ValidOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
            
                
        parser.add_argument('--randomSize', default= False, action='store_false', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        
        parser.add_argument('--resize_or_crop', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', default= True, action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--no_crop', default= True, action='store_true', help='if specified, do not flip the images for data augmentation')
        
        parser.add_argument('--param_path', type=str)
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        
        self.isTrain = True
        return parser