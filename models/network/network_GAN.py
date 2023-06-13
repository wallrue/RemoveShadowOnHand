###############################################################################
# This file contains definitions of GAN and some definitions of RESNET, UNET, 
# UNET Mobile, ResUNet
###############################################################################
import math
import copy
from functools import partial
from beartype import beartype
from torch import einsum
from torch.fft import fft2, ifft2
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.autograd import Variable
from .network_RESNET import resnext101_32x8d
from .network_MobileNet import MobileNetV1, MobileNetV2, mobilenetv3_large, mobilenetv3_small

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
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
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
             n_layers_D=5, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
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

class InvertedResidualBlock(nn.Module):
    """Inverted residual block used in MobileNetV2
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=6, deconvolve=False):
        super(InvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        ex_c = int(self.in_c * expansion_factor)
        if deconvolve:
            self.conv = nn.Sequential(
                # pointwise convolution
                nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.ConvTranspose2d(ex_c, ex_c, 4,self.stride,1, groups=ex_c, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # pointwise convolution
                nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_c),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise convolution
                nn.Conv2d(self.in_c, ex_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.Conv2d(ex_c, ex_c, 3, self.stride, 1, groups=ex_c, bias=False),
                nn.BatchNorm2d(ex_c),
                nn.ReLU6(inplace=True),
                # pointwise convolution
                nn.Conv2d(ex_c, self.out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_c),
            )
        self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

            

    def forward(self, x):
        if self.use_skip_connection:
            out = self.conv(x)
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x+out
        else:
            return self.conv(x)

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
        xx = InvertedResidualBlock(in_c, out_c, s, t, deconvolve=d)
        convs.append(xx)
        if n>1:
            for i in range(1,n):
                xx = InvertedResidualBlock(out_c, out_c, 1, t, deconvolve=d)
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
        # Right arm / Decoding arm with skip connections
        d1 = self.D_irb1(x9) + x6
        d2 = self.D_irb2(d1) + x4
        d3 = self.D_irb3(d2) + x3
        d4 = self.D_irb4(d3) + x2
        d5 = self.DConv4x4(d4)
        out = self.conv1x1_decode(d5)
        return out
    


###############################################################################
# Definitions of ResUNet
###############################################################################

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class LayerNorm(nn.Module):
    def __init__(self, dim, bias = False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock_UNet(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, inner_dim, 1),
        nn.GELU(),
        nn.Conv2d(inner_dim, dim, 1),
    )

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 4,
        depth = 1
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, dim_head = dim_head, heads = heads)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# conditioning class

class Conditioning(nn.Module):
    def __init__(self, fmap_size, dim):
        super().__init__()
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        self.norm_input = LayerNorm(dim, bias = True)
        self.norm_condition = LayerNorm(dim, bias = True)

        self.block = ResnetBlock_UNet(dim, dim)

    def forward(self, x, c):

        # ff-parser in the paper, for modulating out the high frequencies

        dtype = x.dtype
        x = fft2(x)
        x = x * self.ff_parser_attn_map
        x = ifft2(x).real
        x = x.type(dtype)

        # eq 3 in paper

        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # add an extra block to allow for more integration of information
        # there is a downsample right after the Condition block (but maybe theres a better place to condition than right before the downsample)

        return self.block(c)

@beartype
class ResUNet(nn.Module):
    def __init__(
        self,
        dim = 64,
        image_size = 224,
        mask_channels = 1,
        input_img_channels = 3,
        init_dim = None,
        out_dim = None,
        dim_mults: tuple = (1, 2, 4, 8),
        full_self_attn: tuple = (False, False, False, True),
        attn_dim_head = 32,
        attn_heads = 4,
        mid_transformer_depth = 1,
        self_condition = False,
        resnet_block_groups = 8,
        conditioning_klass = Conditioning,
        skip_connect_condition_fmaps = False   # whether to concatenate the conditioning fmaps in the latter decoder upsampling portion of unet
    ):
        super().__init__()

        self.image_size = image_size

        # determine dimensions
        self.input_img_channels = input_img_channels
        self.mask_channels = mask_channels
        self.self_condition = self_condition

        output_channels = mask_channels
        mask_channels = mask_channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding = 3)
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock_UNet, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention related params

        attn_kwargs = dict(
            dim_head = attn_dim_head,
            heads = attn_heads
        )

        # layers

        num_resolutions = len(in_out)
        assert len(full_self_attn) == num_resolutions

        self.conditioners = nn.ModuleList([])

        self.skip_connect_condition_fmaps = skip_connect_condition_fmaps

        # downsampling encoding blocks

        self.downs = nn.ModuleList([])

        curr_fmap_size = image_size

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            self.conditioners.append(conditioning_klass(curr_fmap_size, dim_in))


            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(attn_klass(dim_in, **attn_kwargs)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

            if not is_last:
                curr_fmap_size //= 2

        # middle blocks

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_transformer = Transformer(mid_dim, depth = mid_transformer_depth, **attn_kwargs)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # condition encoding path will be the same as the main encoding path

        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        # upsampling decoding blocks

        self.ups = nn.ModuleList([])

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention

            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_fmaps else 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim = time_dim),
                Residual(attn_klass(dim_out, **attn_kwargs)),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # projection out to predictions

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    def forward(
        self,
        x,
        time,
        cond,
        x_self_cond = None
    ):
        skip_connect_c = self.skip_connect_condition_fmaps
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        c = self.cond_init_conv(cond)

        t = self.time_mlp(time)

        h = []

        for (block1, block2, attn, downsample), (cond_block1, cond_block2, cond_attn, cond_downsample), conditioner in zip(self.downs, self.cond_downs, self.conditioners):
            x = block1(x, t)
            c = cond_block1(c, t)

            h.append([x, c] if skip_connect_c else [x])

            x = block2(x, t)
            c = cond_block2(c, t)

            x = attn(x)
            c = cond_attn(c)

            # condition using modulation of fourier frequencies with attentive map
            # you can test your own conditioners by passing in a different conditioner_klass , if you believe you can best the paper

            c = conditioner(x, c)

            h.append([x, c] if skip_connect_c else [x])

            x = downsample(x)
            c = cond_downsample(c)

        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)

        x = x + c  # seems like they summed the encoded condition to the encoded input representation

        x = self.mid_transformer(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, *h.pop()), dim = 1)
            x = block1(x, t)
            
            x = torch.cat((x, *h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
            
        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)

        return self.final_conv(x)