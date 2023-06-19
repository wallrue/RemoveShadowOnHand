#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict, dataset_dir, checkpoints_dir in train.py to choose the type of experiment and models to train
validDataset_split=0.001
batch_size=4 #8
niter=40
niter_decay=40
epoch_count=1
loadSize=256
fineSize=256
gpu_ids=0

CMD="python ../train.py \
    --loadSize ${loadSize} \
    --fineSize ${fineSize} \
    --batch_size ${batch_size} \
    --gpu_ids ${gpu_ids} \
    --validDataset_split ${validDataset_split}\
    --niter ${niter} \
    --niter_decay ${niter_decay} \
    --epoch_count ${epoch_count}
    "
    
    #--use_skinmask \
    #--continue_train \
    
echo $CMD
eval $CMD

