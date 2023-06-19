###############################################################################
# This file contains class of the dataset named ShadowParam which includes 
# full shadow images, masks of shadow, free shadow images and params of shadow 
###############################################################################

import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset
from data.transform import get_transform_list
from data.image_folder import make_dataset
import cv2

class NTUSTHSDataset(BaseDataset):
    def name(self):
        return 'NTUSTHSDataset'
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'shadowfull')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'shadowfree')
        
        self.A_paths, self.imname = make_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        
        self.transformData = transforms.Compose(get_transform_list(opt))
     
    def __getitem__(self,index):
        birdy = dict()
        
        index_A = index % self.A_size
        imname = self.imname[index_A]
        A_path = self.A_paths[index_A]
        
        A_img = self.transformData(Image.open(A_path).convert('RGB'))      
        C_img = self.transformData(Image.open(os.path.join(self.dir_C, imname)).convert('RGB'))
        
        A_img_num = (np.transpose(A_img.numpy(), (1,2,0)) + 1.0)/2.0 
        A_img_Y, A_img_Cr = self.GetSkinMask(A_img_num)
        A_img_Y, A_img_Cr = torch.from_numpy(A_img_Y*2.0 -1.0), torch.from_numpy(A_img_Cr*2.0 -1.0)
        image_size = A_img.size()
        # Finishing package of dataset information 
        birdy['shadowfull'] = A_img
        birdy['shadowfull_Y'] = torch.permute(A_img_Y, (2, 0, 1))
        birdy['shadowfull_Cr'] = torch.permute(A_img_Cr, (2, 0, 1))
        birdy['shadowfree'] = C_img
        birdy['imgname'] = imname
        birdy['w'] = image_size[0]
        birdy['h'] = image_size[1]
        birdy['shadowfull_paths'] = A_path
        
        return birdy 
    
    def GetSkinMask(self, num_img): #Tensor (3 channels) in range [0, 1]   
        img = np.uint8(num_img*255)
        # Skin color range for hsv color space 
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        img_Y, img_Cr = np.expand_dims(img_YCrCb[:,:,0]/255, axis=2), np.expand_dims(img_YCrCb[:,:,1]/255, axis=2)
        return img_Y, img_Cr
    
    def __len__(self):
        return self.A_size
