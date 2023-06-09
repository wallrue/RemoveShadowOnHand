###############################################################################
# This file is used for training models. 
# There are three major parameters in main function which needs being defined: 
# - dataset_dir: the root folder which contains dataset (with the datasetname)
# - checkpoints_dir: the folder to save checkpoints after training
# - training_dict: the model to be trained (with the datasetname)
# dataset is trained with models in training_dict in a run
###############################################################################

import sys
import os
import ast
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from options.train_options import TrainOptions
from data import CustomDatasetDataLoader
from models import create_model
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
        print("\r{} [{}{}] {}/{} | {:.3f} secs".format(info_dict["epoch"], "#"*x, "."*(size-x), n, count, taken_time), 
                end='', file=out, flush=True) # Flushing for progressing bar in Python 3.0 
        sys.stdout.flush() # Flushing for progressing bar in Python 2.0 
        
    show(0, 1)
    for i, item in enumerate(it):
        yield i, item
        if i == 0: # Initialize batch_size value
            batch_size = len(list(item.values())[0])
        show(i+1, batch_size)
    print("", flush=True, file=out) # Do thing after ending iteration
    
def print_current_losses(log_dir, epoch, lr, iters, losses, t_comp, t_data):
    """ Print current losses on console and save the losses to the disk

    Parameters:
        log_dir (string) -- folder to save log file
        epoch (int) -- current epoch
        lr (float) -- current learning rate
        iters (int) -- current training iteration
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time
        t_data (float) -- data loading time
    """
    message = '{\"epoch\": %d, \"iters\": %d, \"lr\": %.6f, \"computing time\": %.3f, \"data_load_time\": %.3f ' % (epoch, iters, lr, t_comp, t_data)
    for k, v in losses.items():
        message += ', \"%s\": %.3f ' % (k, v)
    message += '}'

    print(" - " + log_dir[-9:] + " : " + message)  # print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
def get_loss_file(log_dir): 
    """ Get current losses from the saving file

    Parameters:
        log_dir (string) -- folder which contains log file
        loss (dict) -- the output dict coming with 'train_loss' and 'valid_loss' in training time
    """
    with open(log_dir, "r+") as f:
        data = f.readlines()
    loss = {'train_loss': list(), 'valid_loss': list()}
    for i in data:
        my_dict = ast.literal_eval(str(i))
        loss['train_loss'].append(my_dict['train_reconstruction'])
        loss['valid_loss'].append(my_dict['valid_reconstruction'])
    return loss

def loss_figure(loss_folder):
    """ Draw loss figure from valid.log file

    Parameters:
        loss_folder (string) -- folder which contains valid.log file
    """
    history = get_loss_file(os.path.join(loss_folder, 'valid.log'))
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    fig, ax = plt.subplots(ncols = 1, nrows = 2, figsize = (20,10))
    N = 1
    ax[0].plot(running_mean(history['train_loss'], N), 'r-', label='training loss')
    ax[1].plot(running_mean(history['valid_loss'], N), 'b-', label='validation loss')
    for i in ax:
        i.legend()
        i.grid(True)
    fig.savefig(os.path.join(loss_folder, 'loss_figure.png'), dpi=100)
    
def train_loop(opt, model): #dataset, model):
    """ This function is training execution plan for dataset, model

    Parameters:
        dataset (string) -- dataset which is used for training
        model (string) -- model which is trained
    """
    cuda_tensor = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 else torch.FloatTensor
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        # Dataset loading
        data_loader = CustomDatasetDataLoader(opt)
        dataset = data_loader.load_data()
        
        epoch_start_time = time.time()
        epoch_iter = 0
        t_comp, t_data = 0, 0
        train_losses = dict()

        dataset.working_subset = "main"
        progressbar_info = {"epoch": "epoch {}/{} ".format(epoch, opt.niter + opt.niter_decay), 
                            "start time": epoch_start_time}
        for i, data in progressbar(dataset, progressbar_info):
            iter_start_time = time.time() # Finishing loading data
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            t_comp += time.time() - iter_start_time # Computing time is from finishing loading data to now

            current_losses = model.get_current_losses()
            train_losses = {key: train_losses.get(key,0) + current_losses[key] for key in set(train_losses).union(current_losses)}
        n_train_losses = epoch_iter/opt.batch_size
        train_losses = {'train_reconstruction': train_losses["G2_L1"]/n_train_losses}
        current_lr = model.update_learning_rate()
        t_data = time.time() - epoch_start_time - t_comp
        print_current_losses(os.path.join(opt.checkpoints_dir, opt.name, 'train.log'), epoch, current_lr, \
                                 epoch_iter, train_losses, t_comp, t_data)
        
        # Validation section
        with torch.no_grad():
            if opt.validDataset_split > 0.0:
                valid_losses, val_loss_phase1, n_valid_loss = 0, 0, 0
                dataset.working_subset = "valid"
                assert len(dataset) > 0, "valid dataset is empty, please change opt.validDataset_split"
                for valid_id, data in enumerate(dataset, 0):
                    full_shadow_img = Variable(data['shadowfull'].type(cuda_tensor))
                    shadow_mask = Variable(data['shadowmask'].type(cuda_tensor))
                    shadowfree_img = Variable(data['shadowfree'].type(cuda_tensor))
                    skin_mask = (Variable(data['skinmask'].type(cuda_tensor)) >0)*2-1
                    
                    if opt.use_skinmask:
                        output = model.get_prediction(full_shadow_img, skin_mask)   
                    else:
                        output = model.get_prediction(full_shadow_img)    
                        
                    val_loss_phase1 += model.criterionL1(output['phase1'], shadow_mask) # Another loss (shadow detect, hand segment,...)
                    valid_losses += model.criterionL1(output['final'], shadowfree_img)
                    n_valid_loss += 1   
            else:
                valid_losses, val_loss_phase1, n_valid_loss = 0, 0, -1
            total_losses = {"valid_reconstruction": valid_losses/ n_valid_loss, 
                            "valid_phase1": val_loss_phase1/ n_valid_loss, **train_losses} #merging 2 dicts
            print_current_losses(os.path.join(opt.checkpoints_dir, opt.name, 'valid.log'), epoch, current_lr, \
                                 0, total_losses, -1.0, -1.0)
           
        # Saving model
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)
        
if __name__=='__main__':
    """The main function for training model. There are three important parameters:
    - dataset_dir: the root folder which contains dataset. {datasetname: path}
    - checkpoints_dir: the folder to save checkpoints after training. {datasetname: path}
    - training_dict: the model to be trained {datasetname: modelname}
    Example of datasetname: shadowparam, rawsynthetic
    Example of modelname: STGAN, SIDSTGAN, SIDPAMIwISTGAN, DSDSID, MedSegDiff
    Example of net_id: lisg of net_id is defined in base_options
    """
    checkpoint_dir = os.path.join(os.getcwd(),"_checkpoints")    
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    train_options = TrainOptions()
    dataset_dir = {"rawsynthetic": os.path.join(os.path.join(os.getcwd(),"_database"),"data_creating"),
                   "shadowparam": os.path.join(os.path.join(os.getcwd(),"_database"),"NTUST_HS_SYNTHETIC"),
                   "shadowsynthetic": os.path.join(os.path.join(os.getcwd(),"_database"),"SYNTHETIC_HAND"),
                   }
    checkpoints_dir = {"rawsynthetic": checkpoint_dir,
                       "shadowparam": checkpoint_dir,
                       "shadowsynthetic": checkpoint_dir
                       }
    
    """ DEFINE EXPERIMENT """
    BACKBONE_TEST = False
    
    # Experient 1: Test the performance of backbones on "shadowparam" dataset
    if BACKBONE_TEST:
        model_name_list = ["STGAN", "SIDSTGAN", "SIDPAMIwISTGAN"]
        training_dict = list()
        for model_name in model_name_list:
            if model_name == "STGAN":        
                training_list = [["shadowparam", model_name, [[i_netG, i_netD],[i_netG, i_netD]], False] for i_netG in range(4) for i_netD in range(2)]
            else: 
                i_netG, i_netD = 0, 0 #The best result from testing "STGAN"
                training_list = [["shadowparam", model_name, [[i_netG, i_netD],[i_netS, i_netG]], False] for i_netS in range(7)] 
            training_dict = training_dict + training_list
    # Experient 2: Test the best backbone from experiment 1 in "shadowsynthetic" dataset
    else:
        #training_dict = [[training_dataset, training_model, backbone, use_skinmask]]
        training_dict =[["rawsynthetic",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], True],
                        ["rawsynthetic",   "SIDSTGAN",         [[0, 0], [5, 0]], True],
                        ["rawsynthetic",   "STGAN",            [[0, 0], [0, 0]], True],
                        ["rawsynthetic",   "SIDPAMIwISTGAN",   [[0, 0], [5, 0]], False],
                        ["rawsynthetic",   "SIDSTGAN",         [[0, 0], [5, 0]], False],
                        ["rawsynthetic",   "STGAN",            [[0, 0], [0, 0]], False],
                        
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
                        
                        # ["shadowsynthetic",   "DSDSID",           [[], [5, 0]], False],  # training DSD requires dataset having hand mask 
                        # ["shadowsynthetic",   "MedSegDiff",       [[], [5, 0]], False],
                        # ["rawsynthetic",   "DSDSID",           [[], [5, 0]], False], 
                        # ["rawsynthetic",   "MedSegDiff",       [[], [5, 0]], False],
                        ]

    """ RUN SECTION """
    for dataset_name, model_name, netid_list, use_skinmask in training_dict:
        print('============== Start training: dataset {}, model {} =============='.format(model_name, dataset_name))  
        train_options.net1_id, train_options.net2_id = netid_list  #backbone for STCGAN, SPM, SPMI Net which was tested above
        train_options.dataset_mode = dataset_name
        train_options.data_root = dataset_dir[dataset_name]
        train_options.checkpoints_root = checkpoints_dir[dataset_name]        
        train_options.model_name = model_name
        opt = train_options.parse()
        
        opt.use_skinmask = use_skinmask
        if opt.use_skinmask:
            opt.name = opt.name + "_HandSeg"
        train_options.print_options(opt)
        
        # Dataset loading
        #data_loader = CustomDatasetDataLoader(opt)
        #dataset = data_loader.load_data()

        # Model defination
        model = create_model(opt)
        model.setup(opt)
    
        # Training
        train_loop(opt, model) #dataset, model)
        loss_figure(os.path.join(opt.checkpoints_dir, opt.name))