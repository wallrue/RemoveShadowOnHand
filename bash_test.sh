#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict, dataset_dir, checkpoints_dir in train.py to choose models to test
use_skinmask=true
epoch_checkpoint_load='latest'
loadSize=256
fineSize=256
batch_size=8
gpu_ids=0

CMD="python test.py \
    --loadSize ${loadSize} \
    --fineSize ${fineSize} \
    --batch_size ${batch_size} \
    --gpu_ids ${gpu_ids} \
    --epoch ${epoch_checkpoint_load}
    "
CMD1="--use_skinmask"

if $use_skinmask
then
  c="${CMD} ${CMD1}"
else
  c="${CMD}"
fi

echo $c
eval $c

