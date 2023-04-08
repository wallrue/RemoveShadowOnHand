import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image,ImageChops
from PIL import ImageFilter
import torch
from pdb import set_trace as st
import random
import numpy as np
import time
class ShadowParamDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_param = os.path.join(opt.dataroot, opt.phase + 'param')
        #self.dir_param = opt.param_path
        self.dir_matte = os.path.join(opt.dataroot, 'matte')
        
        self.A_paths, self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        self.B_size = self.A_size
        
#         transform_list = [transforms.ToTensor(),
#                           transforms.Normalize(mean=opt.norm_mean,
#                                                std = opt.norm_std)]

        self.transform = get_transform(opt) #transforms.Compose(transform_list)
        #self.transformB = transforms.Compose([transforms.ToTensor()])
     
    def __getitem__(self,index):
        birdy = {}
        
        # load A_path, B_path
        index_A = index % self.A_size
        imname = self.imname[index_A]
        A_path = self.A_paths[index_A]
        B_path = os.path.join(self.dir_B, imname.replace('.jpg','.png')) 
        if not os.path.isfile(B_path):
            B_path = os.path.join(self.dir_B, imname)
            print('MASK NOT FOUND : %s'%(B_path))
        
        # load A_img, B_img
        A_img = Image.open(A_path).convert('RGB')        
        ow, oh = A_img.size[0], A_img.size[1]
        w, h = np.float(A_img.size[0]), np.float(A_img.size[1])
        B_img = Image.open(B_path) if os.path.isfile(B_path) else Image.fromarray(np.zeros((int(w),int(h)),dtype = np.float),mode='L')
        C_img = Image.open(os.path.join(self.dir_C, imname)).convert('RGB')
        
        # load shadow_param
        sparam = open(os.path.join(self.dir_param,imname+'.txt'))
        line = sparam.read()
        shadow_param = np.asarray([float(i) for i in line.split(" ") if i.strip()])
        shadow_param = shadow_param[0:6]
        
        
        # Define loadSize (scaled image to this size) -------------------------
#         loadSize = self.opt.loadSize
#         if self.opt.randomSize:
#             loadSize = np.random.randint(loadSize + 1,loadSize * 1.3 ,1)[0]
#         if self.opt.keep_ratio:
#             if w>h:
#                 ratio = np.float(loadSize)/np.float(h)
#                 neww = np.int(w*ratio)
#                 newh = loadSize
#             else:
#                 ratio = np.float(loadSize)/np.float(w)
#                 neww = loadSize
#                 newh = np.int(h*ratio)
#         else:
#             neww = loadSize
#             newh = loadSize
                    
        # Rotate in random in range 90,180 and range -20,20 -------------------------
#         t =[Image.FLIP_LEFT_RIGHT,Image.ROTATE_90]
#         for i in range(0,4):
#             c = np.random.randint(0,3,1,dtype=np.int)[0]
#             if c==2: continue
#             for i in ['A','B','C']:
#                 if i in birdy:
#                     birdy[i]=birdy[i].transpose(t[c])
                    
#         degree=np.random.randint(-20,20,1)[0]
#         for i in ['A','B','C']:
#             birdy[i]=birdy[i].rotate(degree)
        
        # Resize to loadSize -------------------------
#         for k,im in birdy.items():
#             birdy[k] = im.resize((neww, newh),Image.NEAREST)
        
        #w = birdy['A'].size[0]
        #h = birdy['A'].size[1]
        #rescale A to the range [1; log_scale] then take log
        #birdy['penumbra'] = ImageChops.subtract(birdy['B'].filter(ImageFilter.MaxFilter(11)),birdy['B'].filter(ImageFilter.MinFilter(11))) 

        # Transfrom to Tensor and value in range 0 to 1 -------------------------
#         for k,im in birdy.items():
#             birdy[k] = self.transformB(im)
#         for i in ['A','C','B']:
#             if i in birdy:
#                 birdy[i] = (birdy[i] - 0.5)*2         
        
        # Crop image -------------------------
#         if not self.opt.no_crop:        
#             w_offset = random.randint(0,max(0,w-self.opt.fineSize-1))
#             h_offset = random.randint(0,max(0,h-self.opt.fineSize-1))
#             for k,im in birdy.items():   
#                 birdy[k] = im[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        
        # Flip image -------------------------
#         if (not self.opt.no_flip) and random.random() < 0.5:
#             idx = [i for i in range(birdy['A'].size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             for k,im in birdy.items():
#                 birdy[k] = im.index_select(2, idx)
#         for k,im in birdy.items():
#             birdy[k] = im.type(torch.FloatTensor)
            
        birdy['A'] = A_img
        birdy['B'] = B_img
        birdy['C'] = C_img
        for k,im in birdy.items():
            birdy[k] = self.transform(im)
        
        birdy['imname'] = imname
        birdy['w'] = ow
        birdy['h'] = oh
        birdy['A_paths'] = A_path
        birdy['B_baths'] = B_path
        
        #if the shadow area is too small, let's not change anything:
        if torch.sum(birdy['B']>0) < 30 :
            shadow_param=[0,1,0,1,0,1]
        birdy['param'] = torch.FloatTensor(np.array(shadow_param))
        
        return birdy 
    
    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'ShadowParamDataset'
