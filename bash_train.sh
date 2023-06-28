#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict, dataset_dir, checkpoints_dir in train.py to choose the type of experiment and models to train
use_skinmask=false
continue_train=true
epoch_count_start=81
epoch_checkpoint_load='80'
niter=50
niter_decay=50
validDataset_split=0.0
batch_size=8 #batch_size = 4 for DSD Net
loadSize=256
fineSize=256
gpu_ids=0

CMD="python train.py \
    --loadSize ${loadSize} \
    --fineSize ${fineSize} \
    --batch_size ${batch_size} \
    --gpu_ids ${gpu_ids} \
    --validDataset_split ${validDataset_split}\
    --niter ${niter} \
    --niter_decay ${niter_decay} \
    --epoch_count ${epoch_count_start}\
    --epoch ${epoch_checkpoint_load}
    "

CMD1="--use_skinmask"
CMD2="--continue_train"


if $use_skinmask
then
  if $continue_train
  then
    c="${CMD} ${CMD1} ${CMD2}"
  else
    c="${CMD} ${CMD1}"
  fi
else
  if $continue_train
  then
    c="${CMD} ${CMD2}"
  else
    c="${CMD}"
  fi
fi
 
echo $c
eval $c

