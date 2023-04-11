import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch

class SingleDataset(BaseDataset):
    def __init__(self, dataroot,opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')    

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
        if not os.path.isfile(B_path): # change file extension if be not found
            B_path=B_path[:-4]+'.png'
        B_img = Image.open(B_path).convert('L')
           
        ow = A_img.size[0]
        oh = A_img.size[1]
        fineSize = self.opt.fineSize if hasattr(self.opt,'fineSize') else 256
        
        A_img = A_img.resize((fineSize,fineSize))
        B_img = B_img.resize((fineSize,fineSize))
        
        # change into tensor type
        A_img = torch.from_numpy(np.asarray(A_img,np.float32).transpose(2,0,1)).div(255)
        B_img = self.transformB(B_img)
        
        # change from [0, 1] to [-1, 1]
        A_img = A_img*2-1
        B_img = B_img*2-1
        
        # expand dimension
        A_img = A_img.unsqueeze(0)
        B_img = B_img.unsqueeze(0)
        B_img = (B_img>0.2).type(torch.float)*2-1
        return {'A': A_img, 'B':B_img, 'A_paths': A_path, 'imname':imname, 'w':ow, 'h':oh}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
