from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import CustomDatasetDataLoader
from models import create_model
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
        print("\r{} [{}{}] {}/{} | {:.3f} secs".format(info_dict["epoch"], "#"*x, "."*(size-x), n, count, taken_time), 
                end='', file=out, flush=True)
    show(0, 1)
    for i, item in enumerate(it):
        yield i, item
        batch_size = len(list(item.values())[0])
        show(i+1, batch_size)
    print("", flush=True, file=out) #Do thing after ending iteration
    
def print_current_losses(log_dir, epoch, lr, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '{\"epoch\": %d, \"iters\": %d, \"lr\": %.6f, \"computing time\": %.3f, \"data_load_time\": %.3f ' % (epoch, iters, lr, t_comp, t_data)
    for k, v in losses.items():
        message += ', \"%s\": %.3f ' % (k, v)
    message += '}'

    print(" - " + log_dir[-9:] + " : " + message)  # print the message
    with open(log_dir, "a+") as log_file:
        log_file.write('%s\n' % message)  # save the message
        
def get_loss_file(log_dir):            
    with open(log_dir, "r+") as f:
        data = f.readlines()
        
    loss = {'train_loss': list(), 'valid_loss': list()}
    for i in data:
        my_dict = ast.literal_eval(str(i))
        loss['train_loss'].append(my_dict['train_reconstruction'])
        loss['valid_loss'].append(my_dict['valid_reconstruction'])
    return loss

def loss_figure(loss_folder):   
    history = get_loss_file(os.path.join(loss_folder, 'valid.log'))
    #after the training loop returns, we can plot the data
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
    
def train_loop(opt, dataset, model): 
    cuda_tensor = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 else torch.FloatTensor
    for epoch in range(opt.epoch_count, 3): #opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        #model.epoch = epoch
        epoch_iter = 0
        t_comp, t_data = 0, 0
        train_losses = dict()

        dataset.working_subset = "main"
        progressbar_info = {"epoch": "epoch {}/{} ".format(epoch, opt.niter + opt.niter_decay), 
                            "start time": epoch_start_time}
        for i, data in progressbar(dataset, progressbar_info):
            iter_start_time = time.time() #the time after loading data
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            #model.cepoch=epoch
            t_comp += time.time() - iter_start_time #computing time after loading data

            current_losses = model.get_current_losses()
            train_losses = {key: train_losses.get(key,0) + current_losses[key] for key in set(train_losses).union(current_losses)}

        n_train_losses = epoch_iter/opt.batch_size
        train_losses = {'train_reconstruction': train_losses["G2_L1"]/n_train_losses}
        current_lr = model.update_learning_rate()
        t_data = time.time() - epoch_start_time - t_comp
        print_current_losses(os.path.join(opt.checkpoints_dir, opt.name, 'train.log'), epoch, current_lr, \
                                 epoch_iter, train_losses, t_comp, t_data)

        valid_losses, n_valid_loss = 0, 0
        with torch.no_grad():
            dataset.working_subset = "valid"
            for _, data in enumerate(dataset, 0):
                full_shadow_img = Variable(data['shadowfull'].type(cuda_tensor))
                shadow_mask = Variable(data['shadowmask'].type(cuda_tensor))
                shadowfree_img = Variable(data['shadowfree'].type(cuda_tensor))

                output = model.get_prediction(full_shadow_img)        
                #val_loss_G1_L1 = model.criterionL1(output['phase1'], shadow_mask)
                valid_losses += model.criterionL1(output['final'], shadowfree_img)
                n_valid_loss += 1

        valid_losses = valid_losses/ n_valid_loss
        total_losses = {"valid_reconstruction": valid_losses, **train_losses} #merging 2 dicts
        print_current_losses(os.path.join(opt.checkpoints_dir, opt.name, 'valid.log'), epoch, current_lr, \
                             0, total_losses, -1.0, -1.0)
        #saving the model
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)
        
if __name__=='__main__':
    train_options = TrainOptions()
    dataset_dir = {"shadowparam": "C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/data_processing/dataset/SYNTHETIC_HAND/"}
    checkpoints_dir = {"shadowparam": "C:/Users/m1101/Downloads/Shadow_Removal/SID/_Git_SID/"}
    training_dict = [["shadowparam", "SIDSTGAN"]]
    
    for dataset_name, model_name in training_dict:
        print('============== Start training: dataset {}, model {} =============='.format(model_name, dataset_name))
        train_options.dataset_mode = dataset_name
        train_options.data_root = dataset_dir[dataset_name]
        train_options.checkpoints_root = checkpoints_dir[dataset_name]        
        train_options.model_name = model_name
        opt = train_options.parse()
        
        data_loader = CustomDatasetDataLoader(opt) #createDataLoader(opt)
        dataset = data_loader.load_data()

        model = create_model(opt)
        model.setup(opt)

        train_loop(opt, dataset, model)
        loss_figure(os.path.join(opt.checkpoints_dir, opt.name))