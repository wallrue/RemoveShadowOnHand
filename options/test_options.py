from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # data loader argument
        parser.add_argument('--phase', type=str, default='train', help='name of dataset to load. e.g: train, val, test, etc')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # data transform argument
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', default=True, help='if specified, do not flip the images for data augmentation')
 
        # model training configuration
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        

#         parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
#         parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
#         parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
#         parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
#         #  Dropout and Batchnorm has different behavioir during training and test.
#         parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
#         parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

#         parser.set_defaults(model='test')
#         # To avoid cropping, the loadSize should be the same as fineSize
#         parser.set_defaults(loadSize=parser.get_default('fineSize'))
        
        
        self.isTrain = False
        return parser
