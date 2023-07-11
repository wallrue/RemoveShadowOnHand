#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict. Then, if any, dataset_dir, checkpoints_dir in test.py to choose models to test
gpu_ids='0'
epoch_checkpoint_load='latest'
batch_size=4

loadSize=256
fineSize=256

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
$SHELL #prevent bash window from closing
