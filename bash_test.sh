#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict, dataset_dir, checkpoints_dir in train.py to choose models to test
epoch_checkpoint_load=120
batch_size=4
gpu_ids='-1' #not use gpu
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
