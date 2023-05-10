###############################################################################
# This file contains class of the dataset named ShadowParam which only includes 
# full shadow images and masks of shadow 
###############################################################################

import os.path
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset

class SingleDataset(BaseDataset):
    def name(self):
        return 'SingleImageDataset'
    
    def __init__(self, dataroot,opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'shadowfull')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'shadowmask')    

        self.A_paths, self.imname = make_dataset(self.dir_A)
        self.B_paths, _ = make_dataset(self.dir_B)
        
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.imname = sorted(self.imname)
        self.transformB = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        imname = self.imname[index] 
        A_path = os.path.join(self.dir_A, imname)
        B_path = os.path.join(self.dir_B, imname)
        A_img = Image.open(A_path).convert('RGB')
        if not os.path.isfile(B_path): 
            # Change file extension if be not found
            B_path=B_path[:-4]+'.png'
        B_img = Image.open(B_path).convert('L')
           
        ow = A_img.size[0]
        oh = A_img.size[1]
        fineSize = self.opt.fineSize if hasattr(self.opt,'fineSize') else 256
        
        A_img = A_img.resize((fineSize,fineSize))
        B_img = B_img.resize((fineSize,fineSize))
        
        # Change into tensor type
        A_img = torch.from_numpy(np.asarray(A_img,np.float32).transpose(2,0,1)).div(255)
        B_img = self.transformB(B_img)
        
        # Change from [0, 1] to [-1, 1]
        A_img = A_img*2-1
        B_img = B_img*2-1
        
        # Expand dimension
        A_img = A_img.unsqueeze(0)
        B_img = B_img.unsqueeze(0)
        B_img = (B_img>0.2).type(torch.float)*2-1
        return {'shadowfull': A_img, 'shadowmask':B_img, 'shadowfull_paths': A_path, \
                'imgname':imname, 'w':ow, 'h':oh}

    def __len__(self):
        return len(self.A_paths)
