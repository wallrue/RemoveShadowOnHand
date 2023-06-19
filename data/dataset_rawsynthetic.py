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
import random
import cv2
from imutils import paths
from PIL import Image
from scipy.optimize import curve_fit
from data.base_dataset import BaseDataset
from data.transform import get_transform_for_synthetic

class RawSyntheticDataset(BaseDataset):
    def name(self):
        return 'RawSyntheticDataset'
    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_shadowmask = os.path.join(opt.dataroot, 'shadow')
        self.dir_background = os.path.join(opt.dataroot, 'background\\val')
        self.dir_handimg = os.path.join(opt.dataroot, 'hands')
        
        self.shadow_list = list(paths.list_images(self.dir_shadowmask))
        self.background_list = list(paths.list_images(self.dir_background))
        self.handimg_list = list(paths.list_images(self.dir_handimg))
        
        random.shuffle(self.shadow_list)
        random.shuffle(self.background_list)
        random.shuffle(self.handimg_list)
        
        self.transformData_handimg = transforms.Compose(get_transform_for_synthetic(self.opt, 'handimg'))
        self.transformData_background = transforms.Compose(get_transform_for_synthetic(self.opt, 'background'))
        self.transformData_shadow = transforms.Compose(get_transform_for_synthetic(self.opt, 'shadow'))
        
        self.count = 0
        
    def shadow_validator(self, shadow_img, handmask_img):
        valid_score = np.sum(shadow_img)/ np.sum(handmask_img) 
        return valid_score > 0.2 and valid_score < 0.8
    
    def concaten_hand(self, background, norm_mask, object_img): # Concatenate Hands to Background
        # Combine background and hand
        r1 = background[:,:,0]*norm_mask[:,:,0] + np.where(norm_mask[:,:,0] == 0.0, object_img[:,:,0], 0.0)
        r2 = background[:,:,1]*norm_mask[:,:,0] + np.where(norm_mask[:,:,0] == 0.0, object_img[:,:,1], 0.0)
        r3 = background[:,:,2]*norm_mask[:,:,0] + np.where(norm_mask[:,:,0] == 0.0, object_img[:,:,2], 0.0)
        result_img = np.array([r1,r2,r3])
        return np.transpose(result_img, (1,2,0))
    
    def concaten_shadow(self, background, shadow_img): # Concatenate Hands to Background 
        beta = -(random.random() % (80 - 30 + 1) + 30)/100.0; #random from -0.3 to -0.8
        r1 = background[:,:,0] + beta*shadow_img[:,:,0]
        r2 = background[:,:,1] + beta*shadow_img[:,:,0]
        r3 = background[:,:,2] + beta*shadow_img[:,:,0]
        result_img = np.array([r1,r2,r3])
        result_img = np.transpose(result_img, (1,2,0))
        np.clip(result_img, 0.0, 1.0, out=result_img)
        return result_img
    
    def shadow_on_hand(self, hand_mask, shadow_img): # Concatenate Hands to Background
        hand_shadeless = hand_mask*(1.0-shadow_img)
        hand_shaded    = hand_mask*shadow_img
        
        np.clip(hand_shadeless, 0.0, 1.0, out=hand_shadeless)
        np.clip(hand_shaded, 0.0, 1.0, out=hand_shaded)
        return hand_shadeless, hand_shaded
    

    def relit(self, x, a, b): # Functions for computing relit param
        return np.uint8((a * x.astype(np.float64)/255 + b)*255)
    
    def im_relit(self, Rpopt,Gpopt,Bpopt,dump): # Functions for computing relit param
        #some weird bugs with python
        sdim = dump.copy()
        sdim.setflags(write=1)
        sdim = sdim.astype(np.float64)
        sdim[:,:,0] = (sdim[:,:,0]/255) * Rpopt[0] + Rpopt[1]
        sdim[:,:,1] = (sdim[:,:,1]/255) * Gpopt[0] + Gpopt[1]
        sdim[:,:,2] = (sdim[:,:,2]/255) * Bpopt[0] + Bpopt[1]
        sdim = np.uint8(sdim*255)
        return sdim

    def compute_params(self, shadowfull_image, shadowmask_image, shadowfree_image): # Functions for computing relit param
        kernel = np.ones((5,5),np.uint8)
        
        sd = np.uint8(shadowfull_image*255.0)
        mean_sdim = np.mean(sd, axis=2)
        
        mask_ori = np.uint8(shadowmask_image*255.0)
        mask = cv2.erode(mask_ori, kernel, iterations = 2)
        
        sdfree = np.uint8(shadowfree_image*255.0) 
        mean_sdfreeim = np.mean(sdfree, axis=2)

        i, j = np.where(np.logical_and(np.logical_and(np.logical_and(mask>=1,mean_sdim>5),mean_sdfreeim<230),np.abs(mean_sdim-mean_sdfreeim)>10))
        
        source = sd*0
        source[tuple([i,j])] = sd[tuple([i,j])] 
        target = sd*0
        target[tuple([i,j])]= sdfree[tuple([i,j])]
        
        R_s = source[:,:,0][tuple([i,j])]
        G_s = source[:,:,1][tuple([i,j])]
        B_s = source[:,:,2][tuple([i,j])]
        
        R_t = target[:,:,0][tuple([i,j])]
        G_t = target[:,:,1][tuple([i,j])]
        B_t = target[:,:,2][tuple([i,j])]
        
        c_bounds = [[1,-0.1],[10,0.5]]
        
        Rpopt, pcov = curve_fit(self.relit, R_s, R_t, bounds=c_bounds)
        Gpopt, pcov = curve_fit(self.relit, G_s, G_t, bounds=c_bounds)
        Bpopt, pcov = curve_fit(self.relit, B_s, B_t, bounds=c_bounds)
        
        # #Compute error
        # relitim = self.im_relit(Rpopt,Gpopt,Bpopt,sd)
        # error = np.mean(np.abs(relitim[tuple([i,j])].astype(np.float64) - sdfree[tuple([i,j])]).astype(np.float64))
        
        # f = open("C:\\Users\\lemin\\Downloads\\image_test\\params.txt","a")
        # f.write("%f %f %f %f %f %f %f"%(error, Rpopt[1],Rpopt[0],Gpopt[1],Gpopt[0],Bpopt[1],Bpopt[0]))
        # f.close()   
        
        return Rpopt[1],Rpopt[0],Gpopt[1],Gpopt[0],Bpopt[1],Bpopt[0]
        
    def __getitem__(self, index):
        
        birdy = dict()
        index_img = index % len(self.handimg_list)
        handimg = self.transformData_handimg(Image.open(self.handimg_list[index_img]).convert("RGB"))
        background = self.transformData_background(Image.open(self.background_list[index_img]).convert("RGB"))
        
        background_img = (np.transpose(background.numpy(), (1,2,0)) + 1.0)/2.0 
        hand_img = (np.transpose(handimg['img'].numpy(), (1,2,0)) + 1.0)/2.0
        hand_norm = (np.transpose(handimg['binary_normal'].numpy(), (1,2,0)) + 1.0)/2.0
        hand_mask = (np.transpose(handimg['binary_mask'].numpy(), (1,2,0)) + 1.0)/2.0

        # Create shadow mask on hand
        shadow_id = index_img % len(self.shadow_list)
        shadowimg = self.transformData_shadow(Image.open(self.shadow_list[shadow_id]).convert("RGB"))
        shadowimg = (np.transpose(shadowimg.numpy(), (1,2,0)) + 1.0)/2.0
        shadow_img = shadowimg*hand_mask
        while(not self.shadow_validator(shadow_img, hand_mask)):
            shadow_id = (shadow_id+1)%len(self.shadow_list)
            shadowimg = self.transformData_shadow(Image.open(self.shadow_list[shadow_id]).convert("RGB"))
            shadowimg = (np.transpose(shadowimg.numpy(), (1,2,0)) + 1.0)/2.0
            shadow_img = shadowimg*hand_mask
        full_hand_img = self.concaten_hand(background_img, hand_norm, hand_img)
        full_shadow_img = self.concaten_shadow(full_hand_img, shadow_img)

        shadow_param = self.compute_params(full_shadow_img, shadow_img, full_hand_img)  
        skinmask = self.GetSkinMask(full_shadow_img)
        # if self.count < 5: # Save dataset to review
        #     self.count += 1
        #     cv2.imwrite("C:\\Users\\m1101\\Downloads\\image_test\\{}_full_shadow_img.png".format(index), cv2.cvtColor(np.uint8(full_shadow_img*255), cv2.COLOR_RGB2BGR))
        #     cv2.imwrite("C:\\Users\\m1101\\Downloads\\image_test\\{}_shadow_img.png".format(index), np.uint8(shadow_img*255))
        #     cv2.imwrite("C:\\Users\\m1101\\Downloads\\image_test\\{}_full_hand_img.png".format(index), cv2.cvtColor(np.uint8(full_hand_img*255), cv2.COLOR_RGB2BGR))
        #     cv2.imwrite("C:\\Users\\m1101\\Downloads\\image_test\\{}_hand_img.png".format(index), cv2.cvtColor(np.uint8(hand_img*255), cv2.COLOR_RGB2BGR))
        #     cv2.imwrite("C:\\Users\\m1101\\Downloads\\image_test\\{}_hand_mask.png".format(index), np.uint8(hand_mask*255))
        #     cv2.imwrite("C:\\Users\\m1101\\Downloads\\image_test\\{}_skin_mask.png".format(index), np.uint8(skinmask*255))
                        
        shadowfull_image = torch.from_numpy(full_shadow_img*2.0 -1.0)
        shadowmask_image = torch.from_numpy(shadow_img*2.0 -1.0)
        shadowfree_image = torch.from_numpy(full_hand_img*2.0 -1.0)
        handmask_img = torch.from_numpy(hand_mask*2.0 -1.0)
        skinmask_image = torch.from_numpy(skinmask*2.0 -1.0)
     
        image_size = shadowfull_image.size()
        # Finishing package of dataset information    
        birdy['shadowfull'] = torch.permute(shadowfull_image, (2, 0, 1))
        birdy['shadowmask'] = torch.permute(shadowmask_image, (2, 0, 1))
        birdy['shadowfree'] = torch.permute(shadowfree_image, (2, 0, 1))
        birdy['handmask'] = torch.permute(handmask_img, (2, 0, 1))
        birdy['w'] = image_size[0]
        birdy['h'] = image_size[1]

        birdy['shadowparams'] = torch.FloatTensor(np.array(shadow_param))
        birdy['skinmask'] = torch.permute(skinmask_image, (2, 0, 1))
        birdy['imgname'] = "img_{}.png".format(index_img)
        
        return birdy 
    
    def GetSkinMask(self, num_img): #Tensor (3 channels) in range [0, 1]   
        img = np.uint8(num_img*255)

        # Skin color range for hsv color space 
        img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((9,9), np.uint8))
        
        # Skin color range for hsv color space 
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((9,9), np.uint8))
        
        # Merge skin detection (YCbCr and hsv)
        global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
        global_mask=cv2.medianBlur(global_mask,3)
        global_mask = cv2.dilate(global_mask, np.ones((9,9), np.uint8), iterations = 2)
        
        global_mask = np.expand_dims(global_mask/255, axis=2)

        return global_mask
    
    def __len__(self):
        return len(self.handimg_list)
    
    def load_img(self, img_path, size = (224, 224), img_mode = 'L'):
        if os.path.isfile(img_path):
            return Image.open(img_path) 
        else:
            return Image.fromarray(np.zeros((int(size[0]),int(size[1])),dtype = np.float), mode = img_mode)