#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict, dataset_dir, checkpoints_dir in train.py to choose models to test
epoch_checkpoint_load='latest'
loadSize=256
fineSize=256
batch_size=2
gpu_ids=0

CMD="python test.py \
    --loadSize ${loadSize} \
    --fineSize ${fineSize} \
    --batch_size ${batch_size} \
    --gpu_ids ${gpu_ids} \
    --epoch ${epoch_checkpoint_load}
    "
    
c="${CMD}"

echo $c
eval $c

