###############################################################################
# This file defines class for  Distraction-aware Shadow Detection (DSDNet)
# DSDNet is only to detect the shadow, not remove shadow
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn
from .network_RESNET import resnext101_32x4d

def bce_logit_dst(pred, gt):
    """ Loss function for inside layers

    Parameters:
        pred (tensor) -- prediction image
        gt (tensor) -- ground truth
    """
    eposion = 1e-10
    #sigmoid_pred = torch.sigmoid(pred)
    count_pos = torch.sum(gt)*1.0+eposion
    count_neg = torch.sum(1.0-gt)*1.0
    beta = count_neg/count_pos
    beta_back = count_pos / (count_pos + count_neg)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=beta)
    loss = beta_back*bce_loss(pred, gt)
    return loss

def bce_logit_pred(pred, gt, dst1, dst2):
    """ Loss function for output result

    Parameters:
        pred (tensor) -- prediction image
        gt (tensor) -- ground truth
        dst1 (tensor) -- other ground truth (mask of shadow inside hand)
        dst2 (tensor) -- other ground truth (mask of shadow outside hand)
    """
    eposion = 1e-10
    #sigmoid_dst1 = torch.sigmoid(dst1)
    #sigmoid_dst2 = torch.sigmoid(dst2)
    #sigmoid_pred = torch.sigmoid(pred)
    count_pos = torch.sum(gt)*1.0+eposion
    count_neg = torch.sum(1.-gt)*1.0
    beta = count_neg/count_pos
    beta_back = count_pos/(count_pos+count_neg)
    
    dst_loss = beta*(1+dst2)*gt*F.binary_cross_entropy_with_logits(pred, gt, reduction='none') + \
               (1+dst1)*(1-gt)*F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=beta)
    mean_loss = torch.mean(dst_loss)
    loss = beta_back*bce_loss(pred, gt) + beta_back*mean_loss
    return loss

class ConvBlock(nn.Module):
    """ Convolutional Block which is typically used for DSDNet

    Parameters:
        pred (tensor) -- prediction image
        gt (tensor) -- ground truth
        dst1 (tensor) -- other ground truth (mask of shadow inside hand)
        dst2 (tensor) -- other ground truth (mask of shadow outside hand)
    """
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = self.block2(block1)

        return block2

class AttentionModule(nn.Module):
    """ Multi-Attention Block for an image 8x8 (64 pixels). 
    There are 64x64/2 = 64*32 attentions (relations) so we use a conv 64, 32
    """
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(64, 1, 3, bias=False, padding=1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.att(x)
        block2 = block1.repeat(1, 32, 1, 1)

        return block2

class DSDNet(nn.Module):
    def __init__(self):
        super(DSDNet, self).__init__()
        net = resnext101_32x4d(pretrained=False)
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, bias=False, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, bias=False, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.down0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.shad_att = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.dst1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.dst2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.refine4_hl = ConvBlock()
        self.refine3_hl = ConvBlock()
        self.refine2_hl = ConvBlock()
        self.refine1_hl = ConvBlock()
        self.refine0_hl = ConvBlock()

        self.attention4_hl = AttentionModule()
        self.attention3_hl = AttentionModule()
        self.attention2_hl = AttentionModule()
        self.attention1_hl = AttentionModule()
        self.attention0_hl = AttentionModule()
        self.conv1x1_ReLU_down4 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down3 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down2 = nn.Sequential(
            nn.Conv2d(96, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down1 = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.conv1x1_ReLU_down0 = nn.Sequential(
            nn.Conv2d(160, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1, bias=False)
        )

        self.fuse_predict = nn.Sequential(
            nn.Conv2d(5, 1, 1, bias=False)
        )

    def forward(self, x):
        # Backbone
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Encoder -> Image Feature
        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)
        down0 = self.down0(layer0)

        
        down4_shad = down4
        down4_dst1 = self.dst1(down4)
        down4_dst2 = self.dst2(down4)
        
        down4_shad = (1 + self.attention4_hl(torch.cat((down4_shad, down4_dst2), 1))) * down4_shad
        down4_shad = F.relu(-self.refine4_hl(torch.cat((down4_shad, down4_dst1), 1)) + down4_shad, True)
        
        down4_dst1_3 = F.interpolate(down4_dst1,size=down3.size()[2:], mode='bilinear')
        down4_dst1_2 = F.interpolate(down4_dst1,size=down2.size()[2:], mode='bilinear')
        down4_dst1_1 = F.interpolate(down4_dst1,size=down1.size()[2:], mode='bilinear')
        down4_dst1_0 = F.interpolate(down4_dst1,size=down0.size()[2:], mode='bilinear')

        down4_dst2_3 = F.interpolate(down4_dst2,size=down3.size()[2:], mode='bilinear')
        down4_dst2_2 = F.interpolate(down4_dst2,size=down2.size()[2:], mode='bilinear')
        down4_dst2_1 = F.interpolate(down4_dst2,size=down1.size()[2:], mode='bilinear')
        down4_dst2_0 = F.interpolate(down4_dst2,size=down0.size()[2:], mode='bilinear')

        down4_shad_3 = F.interpolate(down4_shad,size=down3.size()[2:], mode='bilinear')
        down4_shad_2 = F.interpolate(down4_shad,size=down2.size()[2:], mode='bilinear')
        down4_shad_1 = F.interpolate(down4_shad,size=down1.size()[2:], mode='bilinear')
        down4_shad_0 = F.interpolate(down4_shad,size=down0.size()[2:], mode='bilinear')
        
        up_down4_dst1 = self.conv1x1_ReLU_down4(down4_dst1)
        up_down4_dst2 = self.conv1x1_ReLU_down4(down4_dst2)
        up_down4_shad = self.conv1x1_ReLU_down4(down4_shad)
        pred_down4_dst1 = F.interpolate(up_down4_dst1,size=x.size()[2:], mode='bilinear')
        pred_down4_dst2 = F.interpolate(up_down4_dst2,size=x.size()[2:], mode='bilinear')
        pred_down4_shad = F.interpolate(up_down4_shad,size=x.size()[2:], mode='bilinear')

        
        down3_dst1 = self.dst1(down3)
        down3_dst2 = self.dst2(down3)
        down3_shad = down3

        down3_shad = (1 + self.attention3_hl(torch.cat((down3_shad, down3_dst2), 1))) * down3_shad
        down3_shad = F.relu(-self.refine3_hl(torch.cat((down3_shad, down3_dst1), 1)) + down3_shad, True)

        down3_dst1_2 = F.interpolate(down3_dst1,size=down2.size()[2:], mode='bilinear')
        down3_dst1_1 = F.interpolate(down3_dst1,size=down1.size()[2:], mode='bilinear')
        down3_dst1_0 = F.interpolate(down3_dst1,size=down0.size()[2:], mode='bilinear')
        down3_dst2_2 = F.interpolate(down3_dst2,size=down2.size()[2:], mode='bilinear')
        down3_dst2_1 = F.interpolate(down3_dst2,size=down1.size()[2:], mode='bilinear')
        down3_dst2_0 = F.interpolate(down3_dst2,size=down0.size()[2:], mode='bilinear')
        down3_shad_2 = F.interpolate(down3_shad,size=down2.size()[2:], mode='bilinear')
        down3_shad_1 = F.interpolate(down3_shad,size=down1.size()[2:], mode='bilinear')
        down3_shad_0 = F.interpolate(down3_shad,size=down0.size()[2:], mode='bilinear')
        up_down3_dst1 = self.conv1x1_ReLU_down3(torch.cat((down3_dst1,down4_dst1_3),1))
        up_down3_dst2 = self.conv1x1_ReLU_down3(torch.cat((down3_dst2,down4_dst2_3),1))
        up_down3_shad = self.conv1x1_ReLU_down3(torch.cat((down3_shad,down4_shad_3),1))
        pred_down3_dst1 = F.interpolate(up_down3_dst1,size=x.size()[2:], mode='bilinear')
        pred_down3_dst2 = F.interpolate(up_down3_dst2,size=x.size()[2:], mode='bilinear')
        pred_down3_shad = F.interpolate(up_down3_shad,size=x.size()[2:], mode='bilinear')


        down2_dst1 = self.dst1(down2)
        down2_dst2 = self.dst2(down2)
        down2_shad = down2
        
        down2_shad = (1 + self.attention2_hl(torch.cat((down2_shad, down2_dst2), 1))) * down2_shad
        down2_shad = F.relu(-self.refine2_hl(torch.cat((down2_shad, down2_dst1), 1)) + down2_shad, True)

        down2_dst1_1 = F.interpolate(down2_dst1,size=down1.size()[2:], mode='bilinear')
        down2_dst1_0 = F.interpolate(down2_dst1,size=down0.size()[2:], mode='bilinear')
        down2_dst2_1 = F.interpolate(down2_dst2,size=down1.size()[2:], mode='bilinear')
        down2_dst2_0 = F.interpolate(down2_dst2,size=down0.size()[2:], mode='bilinear')
        down2_shad_1 = F.interpolate(down2_shad,size=down1.size()[2:], mode='bilinear')
        down2_shad_0 = F.interpolate(down2_shad,size=down0.size()[2:], mode='bilinear')
        up_down2_dst1 = self.conv1x1_ReLU_down2(torch.cat((down2_dst1,down3_dst1_2,down4_dst1_2),1))
        up_down2_dst2 = self.conv1x1_ReLU_down2(torch.cat((down2_dst2,down3_dst2_2,down4_dst2_2),1))
        up_down2_shad = self.conv1x1_ReLU_down2(torch.cat((down2_shad,down3_shad_2,down4_shad_2),1))
        pred_down2_dst1 = F.interpolate(up_down2_dst1,size=x.size()[2:], mode='bilinear')
        pred_down2_dst2 = F.interpolate(up_down2_dst2,size=x.size()[2:], mode='bilinear')
        pred_down2_shad = F.interpolate(up_down2_shad,size=x.size()[2:], mode='bilinear')

        
        down1_dst1 = self.dst1(down1)
        down1_dst2 = self.dst2(down1)
        down1_shad = down1

        down1_shad = (1 + self.attention1_hl(torch.cat((down1_shad, down1_dst2), 1))) * down1_shad
        down1_shad = F.relu(-self.refine1_hl(torch.cat((down1_shad, down1_dst1), 1)) + down1_shad, True)

        down1_dst1_0 = F.interpolate(down1_dst1, size=down0.size()[2:], mode='bilinear')
        down1_dst2_0 = F.interpolate(down1_dst2, size=down0.size()[2:], mode='bilinear')
        down1_shad_0 = F.interpolate(down1_shad, size=down0.size()[2:], mode='bilinear')
        up_down1_dst1 = self.conv1x1_ReLU_down1(torch.cat((down1_dst1,down2_dst1_1,down3_dst1_1,down4_dst1_1),1))
        up_down1_dst2 = self.conv1x1_ReLU_down1(torch.cat((down1_dst2,down2_dst2_1,down3_dst2_1,down4_dst2_1),1))
        up_down1_shad = self.conv1x1_ReLU_down1(torch.cat((down1_shad,down2_shad_1,down3_shad_1,down4_shad_1),1))
        pred_down1_dst1 = F.interpolate(up_down1_dst1,size=x.size()[2:], mode='bilinear')
        pred_down1_dst2 = F.interpolate(up_down1_dst2,size=x.size()[2:], mode='bilinear')
        pred_down1_shad = F.interpolate(up_down1_shad,size=x.size()[2:], mode='bilinear')


        down0_dst1 = self.dst1(down0)
        down0_dst2 = self.dst2(down0)
        down0_shad = down0

        down0_shad = (1 + self.attention0_hl(torch.cat((down0_shad, down0_dst2), 1))) * down0_shad
        down0_shad = F.relu(-self.refine0_hl(torch.cat((down0_shad, down0_dst1), 1)) + down0_shad, True)


        up_down0_dst1 =self.conv1x1_ReLU_down0(torch.cat((down0_dst1,down1_dst1_0,down2_dst1_0,down3_dst1_0,down4_dst1_0),1))
        up_down0_dst2 = self.conv1x1_ReLU_down0(torch.cat((down0_dst2,down1_dst2_0,down2_dst2_0,down3_dst2_0,down4_dst2_0),1))
        up_down0_shad = self.conv1x1_ReLU_down0(torch.cat((down0_shad,down1_shad_0,down2_shad_0,down3_shad_0,down4_shad_0),1))
        pred_down0_dst1 = F.interpolate(up_down0_dst1,size=x.size()[2:], mode='bilinear')
        pred_down0_dst2 = F.interpolate(up_down0_dst2,size=x.size()[2:], mode='bilinear')
        pred_down0_shad = F.interpolate(up_down0_shad,size=x.size()[2:], mode='bilinear')


        fuse_pred_shad = self.fuse_predict(torch.cat((pred_down0_shad,pred_down1_shad,pred_down2_shad,pred_down3_shad,pred_down4_shad),1))
        fuse_pred_dst1 = self.fuse_predict(torch.cat((pred_down0_dst1,pred_down1_dst1,pred_down2_dst1,pred_down3_dst1,pred_down4_dst1),1))
        fuse_pred_dst2 = self.fuse_predict(torch.cat((pred_down0_dst2,pred_down1_dst2,pred_down2_dst2,pred_down3_dst2,pred_down4_dst2),1))

        return fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, \
            fuse_pred_dst1, pred_down1_dst1, pred_down2_dst1, pred_down3_dst1, pred_down4_dst1,\
            fuse_pred_dst2, pred_down1_dst2, pred_down2_dst2, pred_down3_dst2, pred_down4_dst2, \
                   pred_down0_dst1, pred_down0_dst2, pred_down0_shad

def define_DSD(opt):
    net = None
    net = DSDNet()
    if len(opt.gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(opt.gpu_ids[0])
        net = torch.nn.DataParallel(net, opt.gpu_ids)
    return net