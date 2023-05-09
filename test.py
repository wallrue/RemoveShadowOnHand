from options.test_options import TestOptions
from data import CustomDatasetDataLoader
from models import create_model
from models.loss_function import calculate_ssim, calculate_psnr

import sys
import os
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import util.util as util
import matplotlib.pyplot as plt
import time
import ast

def progressbar(it, info_dict, size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j, batch_size):
        n = batch_size*j
        x = int(size*n/count)
        
        taken_time = time.time() - info_dict["start time"]
        print("\r[{}{}] {}/{} | {:.3f} secs".format("#"*x, "."*(size-x), n, count, taken_time), 
                end='', file=out, flush=True)
    show(0, 1)
    for i, item in enumerate(it):
        yield i, item
        batch_size = len(list(item.values())[0])
        show(i+1, batch_size)
    print("", flush=True, file=out) #Do thing after ending iteration
     
def evaluate(dataset, test_model, result_dir):
    cuda_tensor = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 else torch.FloatTensor
    PNSR_dict = {"original": 0.0, "shadowmask": 0.0, "shadowfree": 0.0}
    SSIM_dict = {"original": 0.0, "shadowmask": 0.0, "shadowfree": 0.0}
    
    path_list = [result_dir + "//original", result_dir + "//groudtruth", result_dir + f"//{model_name}"]
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)
    
    tcomp = 0
    progressbar_info = {"start time": time.time()}
    for _, data in progressbar(dataset, progressbar_info):
        list_imgname = data['imgname']
        real_shadowfull = Variable(data['shadowfull'].type(cuda_tensor), 0)
        real_shadowmask = Variable(data['shadowmask'].type(cuda_tensor), 0)
        real_shadowfree = Variable(data['shadowfree'].type(cuda_tensor), 0)

        real_shadowfull = torch.reshape(real_shadowfull,(-1,3,256,256))
        real_shadowmask = torch.reshape(real_shadowmask,(-1,1,256,256))
        real_shadowfree = torch.reshape(real_shadowfree,(-1,3,256,256))
        
        testing_start_time = time.time()         
        prediction = test_model.get_prediction(real_shadowfull)
        tcomp += time.time() - testing_start_time
                 
        fake_shadowfree = prediction['final']
        fake_shadowmask = prediction['phase1']

        # img_fake_B = (shadow_mask + 1.0)/2.0
        # img_fake_B = (img_fake_B - 0.5)*2.0

        # fake_C_ST_GAN = G2_trained(torch.cat((input_img, img_fake_B), 1))
        # RES = test_model.get_prediction(input_img, img_fake_B)
        for i in range(len(real_shadowfull)):
            img_shadowfull = real_shadowfull[i].permute(1, 2, 0)
            img_shadowmask = real_shadowmask[i].permute(1, 2, 0)
            img_shadowfree = real_shadowfree[i].permute(1, 2, 0)
            pre_shadowmask = fake_shadowmask[i].data.permute(1, 2, 0)
            pre_shadowfree = fake_shadowfree[i].data.permute(1, 2, 0)

            PNSR_dict["original"] += calculate_psnr(img_shadowfree, img_shadowfull)
            PNSR_dict["shadowmask"] += calculate_psnr(img_shadowmask, pre_shadowmask)
            PNSR_dict["shadowfree"] += calculate_psnr(img_shadowfree, pre_shadowfree)

            SSIM_dict["original"] += calculate_ssim(img_shadowfree, img_shadowfull)
            SSIM_dict["shadowmask"] += calculate_ssim(img_shadowmask, pre_shadowmask)
            SSIM_dict["shadowfree"] += calculate_ssim(img_shadowfree, pre_shadowfree)

            result_list = list()
            result_list.append(((img_shadowfull + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((img_shadowfree + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((pre_shadowfree + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            
            for idx, path in enumerate(path_list):
                 data_name = path + f"\\{list_imgname[i]}"
                 if not os.path.isfile(data_name):
                    Image.fromarray(result_list[idx]).convert('RGB').resize((224, 224)).save(data_name)
                 
    length = len(dataset)
    PNSR_dict = {k: v / length for k, v in PNSR_dict.items()}
    SSIM_dict = {k: v / length for k, v in SSIM_dict.items()}
    return PNSR_dict, SSIM_dict, tcomp

def print_current_losses(log_dir, model, losses, t_comp):
    message = '{\"testing model\": \"%s\", \"computing time\": %.3f' % (model, t_comp)
    for k, v in losses.items():
        message += ', \"%s\": %s' % (k, v)
    message += '}'

    print(" - Result of testing : " + message)  # print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
if __name__=='__main__':
    test_options = TestOptions()
    dataset_dir = {"shadowsynthetic": "C:\\Users\\m1101\\Downloads\\Shadow_Removal\\SID\\_Git_SID\\data_processing\\dataset\\NTUST_HS"}
    checkpoints_dir = {"shadowsynthetic": "C:\\Users\\m1101\\Downloads\\Shadow_Removal\\SID\\_Git_SID"}
    testing_dict = [["shadowsynthetic", "DSDSID"]]
    result_dir = os.getcwd() + f"\\result_set\\"
    
    for dataset_name, model_name in testing_dict:    
        print('============== Start testing: dataset {}, model {} =============='.format(model_name, dataset_name))
        test_options.dataset_mode = dataset_name
        test_options.data_root = dataset_dir[dataset_name]
        test_options.checkpoints_root = checkpoints_dir[dataset_name]          
        test_options.model_name = model_name
        opt = test_options.parse()
        
        data_loader = CustomDatasetDataLoader(opt) #createDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        model.setup(opt)
        
        PNSR_score, SSIM_score, computing_time = evaluate(dataset, model, result_dir)
        print_current_losses(os.path.join(result_dir, 'valid.log'), model_name, {"PNSR_score": PNSR_score, "SSIM_score": SSIM_score}, computing_time)

