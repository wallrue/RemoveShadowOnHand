###############################################################################
# This file contains class of the dataset named ShadowSynthetic which includes 
# full shadow images, masks of shadow, free shadow images, params of shadow
# hand segmentation images, masks of hand, shadow inside hand images, 
# shadow outside hand images
###############################################################################

import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset
from data.transform import get_transform_list
from data.image_folder import make_dataset

class ShadowSyntheticDataset(BaseDataset):
    def name(self):
        return 'ShadowSyntheticDataset'
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_shadowfull = os.path.join(opt.dataroot, opt.phase + 'shadowfull')
        self.dir_shadowmask = os.path.join(opt.dataroot, opt.phase + 'shadowmask')
        self.dir_shadowfree = os.path.join(opt.dataroot, opt.phase + 'shadowfree')
        self.dir_shadowparams = os.path.join(opt.dataroot, opt.phase + 'shadowparams')
        self.dir_handimg = os.path.join(opt.dataroot, opt.phase + 'handimg')
        self.dir_handmask = os.path.join(opt.dataroot, opt.phase + 'handmask')
        self.dir_handshaded = os.path.join(opt.dataroot, opt.phase + 'handshaded')
        self.dir_handshadedless = os.path.join(opt.dataroot, opt.phase + 'handshadedless')
        #self.dir_matte = os.path.join(opt.dataroot, 'shadowmatte')
        
        self.img_paths, self.imname = make_dataset(self.dir_shadowfull)
        self.img_size = len(self.img_paths)
        self.shadow_size = self.img_size
        
        self.transformData = transforms.Compose(get_transform_list(opt))
        self.transformShadow = transforms.Compose([transforms.ToTensor()])
     
    def __getitem__(self,index):
        birdy = dict()
        
        index_img = index % self.img_size
        imname = self.imname[index_img]
        img_path = self.img_paths[index_img]
        shadow_path = os.path.join(self.dir_shadowmask, imname.replace('.jpg','.png')) 
        handmask_path = os.path.join(self.dir_handmask, imname.replace('.jpg','.png')) 
        handshaded_path = os.path.join(self.dir_handshaded, imname.replace('.jpg','.png')) 
        handshadedless_path = os.path.join(self.dir_handshadedless, imname.replace('.jpg','.png')) 
        
        shadowfull_img = Image.open(img_path).convert('RGB')        
        ow, oh = shadowfull_img.size[0], shadowfull_img.size[1]
        w, h = np.float(shadowfull_img.size[0]), np.float(shadowfull_img.size[1])
        shadowfree_img = Image.open(os.path.join(self.dir_shadowfree, imname)).convert('RGB')
        
        shadowmask_img = self.load_img(shadow_path, (w, h), img_mode = 'L')
        handmask_img = self.load_img(handmask_path, (w, h), img_mode = 'L')
        shandshaded_img = self.load_img(handshaded_path, (w, h), img_mode = 'L')
        handshadedless_img = self.load_img(handshadedless_path, (w, h), img_mode = 'L')
        handimg_img = self.load_img(os.path.join(self.dir_handimg, imname), (w, h), img_mode = 'RGB')
        
        # Load shadow_param
        sparam = open(os.path.join(self.dir_shadowparams,imname+'.txt'))
        line = sparam.read()
        shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()])
        shadow_param = shadow_param[0:6]
        
        # Finishing package of dataset information    
        birdy['shadowfull'] = shadowfull_img
        birdy['shadowmask'] = shadowmask_img
        birdy['shadowfree'] = shadowfree_img
        birdy['handmask'] = handmask_img
        birdy['handshaded'] = shandshaded_img
        birdy['handshadedless'] = handshadedless_img
        birdy['handimg'] = handimg_img
        for k,im in birdy.items():
            birdy[k] = self.transformData(im)
        
        birdy['imgname'] = imname
        birdy['w'] = ow
        birdy['h'] = oh
        birdy['shadowfull_paths'] = img_path
        birdy['shadowmask_baths'] = shadow_path
        
        if torch.sum(birdy['shadowmask']>0) < 30 :
            shadow_param=[0,1,0,1,0,1]
        birdy['shadowparams'] = torch.FloatTensor(np.array(shadow_param))
        return birdy 
    
    def __len__(self):
        return max(self.img_size, self.shadow_size)
    
    def load_img(self, img_path, size = (224, 224), img_mode = 'L'):
        if os.path.isfile(img_path):
            return Image.open(img_path) 
        else:
            return Image.fromarray(np.zeros((int(size[0]),int(size[1])),dtype = np.float), mode = img_mode)