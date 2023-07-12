#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict. Then, if any, dataset_dir, checkpoints_dir in train.py to choose the type of experiment and models to train
gpu_ids='0'
continue_train=false
epoch_count_start=1
epoch_checkpoint_load=$((epoch_count_start - 1))
niter=1
niter_decay=1
batch_size=8 #batch_size = 4 for DSD Net
save_epoch_freq=2

loadSize=256
fineSize=256
validDataset_split=0.0

CMD="python train.py \
    --loadSize ${loadSize} \
    --fineSize ${fineSize} \
    --batch_size ${batch_size} \
    --gpu_ids ${gpu_ids} \
    --save_epoch_freq ${save_epoch_freq} \
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
for VARIABLE in 1 #2 3 4
do
    epoch_checkpoint_load=$((niter + niter_decay))
    epoch_count_start=$((niter + niter_decay + 1))
    niter=$((niter + niter_decay + niter_decay))
    niter_decay=$((niter_decay + 0))
    CMD="python train.py \
        --loadSize ${loadSize} \
        --fineSize ${fineSize} \
        --batch_size ${batch_size} \
        --gpu_ids ${gpu_ids} \
        --save_epoch_freq ${save_epoch_freq} \
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
$SHELL #prevent bash window from closing



