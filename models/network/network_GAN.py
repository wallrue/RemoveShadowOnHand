###############################################################################
# This file contains definitions of GAN and some definitions of RESNET, UNET, 
# UNET Mobile
###############################################################################

import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.autograd import Variable
from .network_RESNET import resnext101_32x8d
from .network_MobileNet import MobileNetV1, InvertedBlockV2, MobileNetV2, mobilenetv3_large, mobilenetv3_small

###############################################################################
# Definitions of GAN
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetModel(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetModel(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetModel(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetModel(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_32':
        net = UnetModel(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'mobile_unet':
        net = MobileUNet(input_nc, output_nc)
    elif netG == 'RESNEXT':
        net = resnext101_32x8d(pretrained=False, num_classes=output_nc, num_inputchannels=input_nc)
    elif netG == 'mobilenetV1':
        net = MobileNetV1(ch_in=input_nc, n_classes=output_nc)
        print('MobileNetV1 should have 3 input channels')
    elif netG == 'mobilenetV2':
        net = MobileNetV2(ch_in=input_nc, n_classes=output_nc)
        print('MobileNetV2 should have 3 input channels')
    elif netG == 'mobilenetV3_large':       
        net = mobilenetv3_large(ch_in=input_nc, ch_out=output_nc)
        print('MobileNetV3 should have 3 input channels')    
    elif netG == 'mobilenetV3_small':
        net = mobilenetv3_small(ch_in=input_nc, ch_out=output_nc)
        print('MobileNetV3 should have 3 input channels')      
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    """ Defines the GAN loss which uses either LSGAN or the regular GAN.
    When LSGAN is used, it is basically same as MSELoss,
    but it abstracts away the need to create the target label tensor
    that has the same size as the input
    """
    def __init__(self, gpu_ids, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.device = torch.device(gpu_ids[0]) if len(gpu_ids) > 0 else torch.device('cpu')
        #self.cuda_tensor = torch.FloatTensor if self.device == torch.device('cpu') else torch.cuda.FloatTensor
        self.register_buffer('real_label', torch.tensor(target_real_label, device=self.device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label, device=self.device))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            #target_tensor = Variable(self.cuda_tensor(torch.ones(input.size())), requires_grad=False)
            target_tensor = Variable(self.real_label.expand_as(input), requires_grad=False)
        else:
            #target_tensor = Variable(self.cuda_tensor(torch.zeros(input.size())), requires_grad=False)
            target_tensor = Variable(self.fake_label.expand_as(input), requires_grad=False)
        return target_tensor #target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

###############################################################################
# Definitions of RESNET
###############################################################################

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    # Sequential block
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        
        # First convolutional part
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
 
        # Second convolutional part
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        
        return nn.Sequential(*conv_block)

    # Residual network
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetModel(nn.Module):
    """ Defines the generator that consists of Resnet blocks between a few
    downsampling/upsampling operations.
    Code and idea originally from Justin Johnson's architecture.
    https://github.com/jcjohnson/fast-neural-style/
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetModel, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        # Normalize for the input channel
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        # Define the downsampling part which shrinks features
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # Study features after downsampling by resnet
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # Define the upsampling part which expands features
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            
        # Normalize for the output channel
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

###############################################################################
# Definitions of UNET
###############################################################################

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost: 
            # The highest layer of UNET archiecture
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost: 
            # The deepest layer of UNET archiecture
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else: 
            # The rest layers
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetModel(nn.Module):
    """ Defines the Unet generator.
    |num_downs|: number of downsamplings in UNet. For example,
    if |num_downs| == 7, image of size 128x128 will become of size 1x1
    at the bottleneck
    
    Defines the submodule with skip connection.
    X -------------------identity---------------------- X
    |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetModel, self).__init__()

        # Middle layers (deepest layers) of unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            
        # Upper layers of unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        
        # Outermost layer (the highest layer) of unet structure
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)
        

class NLayerDiscriminator(nn.Module):
    """ Defines the PatchGAN discriminator with the specified arguments.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        
        # The first convolutional layer
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # The next convolutional layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # The last convolutional layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """ Defines the simple Pixel Discriminator with the specified arguments.
    """
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
    
    
###############################################################################
# Definitions of Mobile UNet
###############################################################################

# class InvertedResidualBlock(nn.Module):
#     """Inverted residual block used in MobileNetV2
#     """
#     def __init__(self, in_c, out_c, stride, expansion_factor=6, deconvolve=False):
#         super(InvertedResidualBlock, self).__init__()
#         # check stride value
#         assert stride in [1, 2]
#         self.stride = stride
#         self.in_c = in_c
#         self.out_c = out_c
#         # Skip connection if stride is 1
#         self.use_skip_connection = True if self.stride == 1 else False

#         # expansion factor or t as mentioned in the paper
#         ex_c = int(self.in_c * expansion_factor)
#         if deconvolve:
#             self.conv = nn.Sequential(
#                 # pointwise convolution
#                 nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(ex_c),
#                 nn.ReLU6(inplace=True),
#                 # depthwise convolution
#                 nn.ConvTranspose2d(ex_c, ex_c, 4,self.stride,1, groups=ex_c, bias=False),
#                 nn.BatchNorm2d(ex_c),
#                 nn.ReLU6(inplace=True),
#                 # pointwise convolution
#                 nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(self.out_c),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pointwise convolution
#                 nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(ex_c),
#                 nn.ReLU6(inplace=True),
#                 # depthwise convolution
#                 nn.Conv2d(ex_c, ex_c, 3, self.stride, 1, groups=ex_c, bias=False),
#                 nn.BatchNorm2d(ex_c),
#                 nn.ReLU6(inplace=True),
#                 # pointwise convolution
#                 nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(self.out_c),
#             )
#         self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

            

#     def forward(self, x):
#         if self.use_skip_connection:
#             out = self.conv(x)
#             if self.in_c != self.out_c:
#                 x = self.conv1x1(x)
#             return x+out
#         else:
#             return self.conv(x)

class MobileUNet(nn.Module):
    """Modified UNet with inverted residual block and depthwise seperable convolution
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=32):
        super(MobileUNet, self).__init__()

        # encoding arm
        self.conv3x3 = self.depthwise_conv(input_nc, 32, p=1, s=2)
        self.irb_bottleneck1 = self.irb_bottleneck(32, 16, 1, 1, 1)
        self.irb_bottleneck2 = self.irb_bottleneck(16, 24, 2, 2, 6)
        self.irb_bottleneck3 = self.irb_bottleneck(24, 32, 3, 2, 6)
        self.irb_bottleneck4 = self.irb_bottleneck(32, 64, 4, 2, 6)
        self.irb_bottleneck5 = self.irb_bottleneck(64, 96, 3, 1, 6)
        self.irb_bottleneck6 = self.irb_bottleneck(96, 160, 3, 2, 6)
        self.irb_bottleneck7 = self.irb_bottleneck(160, 320, 1, 1, 6)
        self.conv1x1_encode = nn.Conv2d(320, 1280, kernel_size=1, stride=1)
        
        # decoding arm
        self.D_irb1 = self.irb_bottleneck(1280, 96, 1, 2, 6, True)
        self.D_irb2 = self.irb_bottleneck(96, 32, 1, 2, 6, True)
        self.D_irb3 = self.irb_bottleneck(32, 24, 1, 2, 6, True)
        self.D_irb4 = self.irb_bottleneck(24, 16, 1, 2, 6, True)
        self.DConv4x4 = nn.ConvTranspose2d(16, 16, 4, 2, 1, groups=16, bias=False)
        
        # Final layer: output channel number can be changed as per the usecase
        self.conv1x1_decode = nn.Conv2d(16, output_nc, kernel_size=1, stride=1)

    def depthwise_conv(self, in_c, out_c, k=3, s=1, p=0):
        """Optimized convolution by combining depthwise convolution and
        pointwise convolution.
        """
        conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, stride=s),
            nn.BatchNorm2d(num_features=in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=1),
        )
        return conv
    
    def irb_bottleneck(self, in_c, out_c, n, s, t, d=False):
        """Create a series of inverted residual blocks.
        """
        convs = []
        xx = InvertedBlockV2(in_c, out_c, t, s) #, deconvolve=d)
        convs.append(xx)
        if n>1:
            for i in range(1,n):
                xx = InvertedBlockV2(out_c, out_c, t, 1) #, deconvolve=d)
                convs.append(xx)
        conv = nn.Sequential(*convs)
        return conv
    
    def get_count(self, model):
        # simple function to get the count of parameters in a model.
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num
    
    def forward(self, x):
        # Left arm/ Encoding arm
        #D1
        x1 = self.conv3x3(x) #(32, 112, 112)
        x2 = self.irb_bottleneck1(x1) #(16,112,112) s1
        x3 = self.irb_bottleneck2(x2) #(24,56,56) s2
        x4 = self.irb_bottleneck3(x3) #(32,28,28) s3
        x5 = self.irb_bottleneck4(x4) #(64,14,14)
        x6 = self.irb_bottleneck5(x5) #(96,14,14) s4
        x7 = self.irb_bottleneck6(x6) #(160,7,7)
        x8 = self.irb_bottleneck7(x7) #(320,7,7)
        x9 = self.conv1x1_encode(x8) #(1280,7,7) s5
        print(x6.shape, x7.shape, x9.shape, self.D_irb1(x9).shape)
        # Right arm / Decoding arm with skip connections
        d1 = self.D_irb1(x9) + x6
        d2 = self.D_irb2(d1) + x4
        d3 = self.D_irb3(d2) + x3
        d4 = self.D_irb4(d3) + x2
        d5 = self.DConv4x4(d4)
        out = self.conv1x1_decode(d5)
        return out
    


