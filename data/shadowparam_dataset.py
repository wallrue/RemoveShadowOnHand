import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.transform import get_transform_list
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np

class ShadowParamDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_param = os.path.join(opt.dataroot, opt.phase + 'train_params')
        self.dir_matte = os.path.join(opt.dataroot, 'matte')
        
        self.A_paths, self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        self.B_size = self.A_size
        
        self.transformData = transforms.Compose(get_transform_list(opt))
        self.transformB = transforms.Compose([transforms.ToTensor()])
     
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
            
        birdy['A'] = A_img
        birdy['B'] = B_img
        birdy['C'] = C_img
        for k,im in birdy.items():
            birdy[k] = self.transformData(im)
        
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
