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
import cv2
from PIL import Image
from torch.autograd import Variable
from options.test_options import TestOptions
from data import CustomDatasetDataLoader
from models import create_model
from models.loss_function import calculate_ssim, calculate_psnr
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #Fix error on computer
  
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
        n = batch_size*j if batch_size*j < count else count
        x = int(size*n/count) 
        
        taken_time = time.time() - info_dict["start time"]
        print("\r[{}{}] {}/{} | {:.3f} secs".format("#"*x, "."*(size-x), n, count, taken_time), 
                end='', file=out, flush=True) # Flushing for progressing bar in Python 3.0 
        sys.stdout.flush() # Flushing for progressing bar in Python 2.0 
        
    show(0, 1)
    for i, item in enumerate(it):
        yield i, item
        if i == 0: # Initialize batch_size value
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
        message += ', \t\"%s\": %s' % (k, v)
    message += '}'

    print(" - Result of testing : " + message)  # Print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # Save the message

def GetSkinMask(tensor_img): #Tensor (4 channels) in range [-1, 1]   
    result_list = list()
    for i in range(len(tensor_img)):
        num_img = tensor_img[i].cpu()
        num_img = (np.transpose(num_img, (1,2,0)) + 1.0)/2.0
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
        result_list.append(global_mask)
    result = torch.from_numpy(np.array(result_list))
    return result

def evaluate(dataset, test_model, result_dir, folder_name):
    """The function is used for assessing the accuracy of test_model on dataset.
    
    Parameters:
        dataset (string) -- dataset for assessing
        test_model (string) -- test_model need to be evaluated
        result_dir (string) -- folder for storing result
    """
    cuda_tensor = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 else torch.FloatTensor
    PNSR_dict = {"original": 0.0, "shadowmask": 0.0, "shadowfree": 0.0}
    SSIM_dict = {"original": 0.0, "shadowmask": 0.0, "shadowfree": 0.0}
    
    path_list = [os.path.join(result_dir,"original"), 
                 os.path.join(result_dir,"groudtruth"),
                 os.path.join(result_dir,"shadowmask_{}".format(folder_name)),  
                 os.path.join(result_dir,"shadowfree_{}".format(folder_name))]
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
        if not opt.use_skinmask:
            prediction = test_model.get_prediction(real_shadowfull)
        else:
            skin_mask = (GetSkinMask(real_shadowfull) >0)*2-1
            skin_mask = torch.reshape(skin_mask,(-1,1,256,256))
            prediction = test_model.get_prediction(real_shadowfull, skin_mask)
        tcomp += time.time() - testing_start_time
                 
        fake_shadowfree = prediction['final']
        fake_shadowmask = prediction['phase1']
        
        for i in range(len(real_shadowfull)):
            img_shadowfull = real_shadowfull[i].permute(1, 2, 0)
            img_shadowmask = real_shadowmask[i][0]
            img_shadowfree = real_shadowfree[i].permute(1, 2, 0)
            pre_shadowmask = fake_shadowmask[i][0].data
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
            result_list.append(((pre_shadowmask + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            result_list.append(((pre_shadowfree + 1.0)*255.0/2.0).cpu().numpy().astype(np.uint8))
            
            # Save output image after processing
            for idx, path in enumerate(path_list):
                data_name = os.path.join(path,"{}".format(list_imgname[i]))
                if not os.path.isfile(data_name):
                    result_img = result_list[idx]
                    if len(np.shape(result_img)) == 3 and np.shape(result_img)[2] == 3:
                        Image.fromarray(result_img).convert('RGB').resize((224, 224)).save(data_name)
                    elif len(np.shape(result_img)) == 2:
                        Image.fromarray(result_img).resize((224, 224)).save(data_name)
      
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
    checkpoint_dir = os.path.join(os.getcwd(), "_checkpoints")      
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    test_options = TestOptions()
    dataset_dir = {"NTUST_HS": os.path.join(os.path.join(os.getcwd(),"_database"),"NTUST_HS_Testset"),
                   "rawsynthetic": os.path.join(os.path.join(os.getcwd(),"_database"),"data_creating"),
                   "shadowparam": os.path.join(os.path.join(os.getcwd(),"_database"),"NTUST_HS_SYNTHETIC"),
                   "shadowsynthetic": os.path.join(os.path.join(os.getcwd(),"_database"),"SYNTHETIC_HAND"),
                   }
    checkpoints_dir = {"rawsynthetic": checkpoint_dir,
                       "shadowparam": checkpoint_dir,
                       "shadowsynthetic": checkpoint_dir
                       }

    testing_dict =[ ["rawsynthetic",   "STGAN",            [[0, 0], [0, 0]], False],
                    ["rawsynthetic",   "SIDSTGAN",         [[0, 0], [5, 0]], False],
                    ["rawsynthetic",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], False],
                    ["rawsynthetic",   "STGAN",            [[0, 0], [0, 0]], True],
                    ["rawsynthetic",   "SIDSTGAN",         [[0, 0], [5, 0]], True],
                    ["rawsynthetic",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], True],
                    
                    # ["shadowparam",   "STGAN",            [[0, 0], [0, 0]], False],
                    # ["shadowparam",   "SIDSTGAN",         [[0, 0], [5, 0]], False],
                    # ["shadowparam",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], False],
                    # ["shadowparam",   "STGAN",            [[0, 0], [0, 0]], True],
                    # ["shadowparam",   "SIDSTGAN",         [[0, 0], [5, 0]], True],
                    # ["shadowparam",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], True],
                    
                    # ["shadowsynthetic",   "STGAN",            [[0, 0], [0, 0]], False],
                    # ["shadowsynthetic",   "SIDSTGAN",         [[0, 0], [5, 0]], False],
                    # ["shadowsynthetic",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], False],
                    # ["shadowsynthetic",   "STGAN",            [[0, 0], [0, 0]], True],
                    # ["shadowsynthetic",   "SIDSTGAN",         [[0, 0], [5, 0]], True],
                    # ["shadowsynthetic",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], True],
                    
                    # ["shadowsynthetic",   "DSDSID",           [[], [5, 0]], False], 
                    # ["shadowsynthetic",   "MedSegDiff",       [[], [5, 0]], False],
                    # ["rawsynthetic",   "DSDSID",           [[], [5, 0]], False],
                    # ["rawsynthetic",   "MedSegDiff",       [[], [5, 0]], False],
                    ]
        
    result_dir = os.path.join(os.getcwd(), "_result_set")
    test_dataset_mode, test_dataset_path = "NTUST_HS", "NTUST_HS"#"shadowsynthetic"
    for dataset_name, model_name, netid_list, use_skinmask in testing_dict:    
        print('============== Start testing: dataset {}, model {} =============='.format(model_name, dataset_name))
        # Model defination   
        test_options.net1_id, test_options.net2_id = netid_list
        test_options.dataset_mode = dataset_name
        test_options.checkpoints_root = checkpoints_dir[dataset_name]        
        test_options.model_name = model_name
        opt = test_options.parse()
        
        opt.use_skinmask = use_skinmask        
        if opt.use_skinmask:
            opt.name = opt.name + "_HandSeg"
        test_options.print_options(opt)
        
        model = create_model(opt)
        model.setup(opt)

        # Loading test dataset
        opt.dataset_mode = test_dataset_mode # Dataset loading -- for "NTUST_HS"
        opt.dataroot = dataset_dir[test_dataset_path]
        data_loader = CustomDatasetDataLoader(opt)
        dataset = data_loader.load_data()

        experiment_name = opt.name #"{}_{}".format(model_name, dataset_name)
        PNSR_score, SSIM_score, computing_time = evaluate(dataset, model, result_dir, experiment_name)
        print_current_losses(os.path.join(result_dir, 'valid.log'), experiment_name, {"PNSR_score": PNSR_score, "SSIM_score": SSIM_score}, computing_time)