###############################################################################
# This file is used for testing trained models. 
# There are three major parameters in main function which needs being defined: 
# - dataset_dir: the root folder which contains dataset (with the datasetname)
# - checkpoints_dir: the folder to save checkpoints after training
# - training_dict: the model to be trained (with the datasetname)
# dataset is trained with models in training_dict in a run
###############################################################################

import sys
import os
import torch
import numpy as np
import time
from PIL import Image
from torch.autograd import Variable
from options.test_options import TestOptions
from data import CustomDatasetDataLoader
from models import create_model
from models.loss_function import calculate_ssim, calculate_psnr

def progressbar(it, info_dict, size=60, out=sys.stdout):
    """The function for displaying progress bar 
    
    Parameters:
        it (int) -- current training iteration
        info_dict (dict) -- information to display (include start time, total epoch)
        size (int) -- length of progress bar
        out (int) -- the saving folder
    """
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
    print("", flush=True, file=out) # Do thing after ending iteration
      
def print_current_losses(log_dir, model, losses, t_comp):
    """ Print current losses on console and save the losses to the disk

    Parameters:
        log_dir (string) -- folder to save log file
        model (string) -- model name
        losses (OrderedDict) -- testing losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time
    """
    message = '{\"testing model\": \"%s\", \"computing time\": %.3f' % (model, t_comp)
    for k, v in losses.items():
        message += ', \"%s\": %s' % (k, v)
    message += '}'

    print(" - Result of testing : " + message)  # Print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # Save the message
        
def evaluate(dataset, test_model, result_dir):
    """The function is used for assessing the accuracy of test_model on dataset.
    
    Parameters:
        dataset (string) -- dataset for assessing
        test_model (string) -- test_model need to be evaluated
        result_dir (string) -- folder for storing result
    """
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
            
            # Save result of processing to list
            result_list = list()
            result_list.append(((img_shadowfull + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((img_shadowfree + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((pre_shadowfree + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            
            # Save output image after processing
            for idx, path in enumerate(path_list):
                data_name = path + f"\\{list_imgname[i]}"
                if not os.path.isfile(data_name):
                    Image.fromarray(result_list[idx]).convert('RGB').resize((224, 224)).save(data_name)
                 
    length = len(dataset)
    PNSR_dict = {k: v / length for k, v in PNSR_dict.items()}
    SSIM_dict = {k: v / length for k, v in SSIM_dict.items()}
    return PNSR_dict, SSIM_dict, tcomp
        
if __name__=='__main__':
    """The main function for testing model. There are three important parameters:
    - dataset_dir: the root folder which contains dataset. {datasetname: path}
    - checkpoints_dir: the folder to save checkpoints after training. {datasetname: path}
    - training_dict: the model to be trained {datasetname: modelname}
    Example of datasetname: shadowparam, shadowsynthetic, single
    Example of modelname: DSDSID, SIDSTGAN, STGAN
    """
    test_options = TestOptions()
    dataset_dir = {"shadowsynthetic": "C:\\Users\\m1101\\Downloads\\Shadow_Removal\\SID\\_Git_SID\\data_processing\\dataset\\NTUST_HS"}
    checkpoints_dir = {"shadowsynthetic": "C:\\Users\\m1101\\Downloads\\Shadow_Removal\\SID\\_Git_SID"}
    testing_dict = [["shadowsynthetic", "DSDSID"]]
    result_dir = os.getcwd() + "\\result_set\\"
    
    for dataset_name, model_name in testing_dict:    
        print('============== Start testing: dataset {}, model {} =============='.format(model_name, dataset_name))
        test_options.dataset_mode = dataset_name
        test_options.data_root = dataset_dir[dataset_name]
        test_options.checkpoints_root = checkpoints_dir[dataset_name]          
        test_options.model_name = model_name
        opt = test_options.parse()
        
        data_loader = CustomDatasetDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)
        model.setup(opt)
        
        PNSR_score, SSIM_score, computing_time = evaluate(dataset, model, result_dir)
        print_current_losses(os.path.join(result_dir, 'valid.log'), model_name, {"PNSR_score": PNSR_score, "SSIM_score": SSIM_score}, computing_time)
