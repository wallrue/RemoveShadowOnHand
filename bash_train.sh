#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict, dataset_dir, checkpoints_dir in train.py to choose the type of experiment and models to train
continue_train=false
epoch_count_start=1
epoch_checkpoint_load='latest'
niter=2
niter_decay=2

validDataset_split=0.0
batch_size=2 #batch_size = 4 for DSD Net
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
CMD1="--continue_train"

if $continue_train
then
  c="${CMD} ${CMD1}"
else
  c="${CMD}"
fi

echo $c
eval $c
for VARIABLE in 1 2
do
    epoch_count_start=$((niter + niter_decay + 1))
    niter=$((niter + 2))
    niter_decay=$((niter_decay + 2))
    
    CMD="python train.py \
        --loadSize ${loadSize} \
        --fineSize ${fineSize} \
        --batch_size ${batch_size} \
        --gpu_ids ${gpu_ids} \
        --validDataset_split ${validDataset_split}\
        --niter ${niter} \
        --niter_decay ${niter_decay} \
        --epoch_count ${epoch_count_start}\
        --epoch ${epoch_checkpoint_load}\
        --continue_train\
        "
    c="${CMD}"
    echo $c
    eval $c
done




