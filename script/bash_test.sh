#!/bin/bash
#NOTE: Modify BACKBONE_TEST, training_dict, dataset_dir, checkpoints_dir in train.py to choose models to test
loadSize=256
fineSize=256
batch_size=1
gpu_ids=0
epoch_checkpoints='80'
CMD="python ../test.py \
    --loadSize ${loadSize} \
    --fineSize ${fineSize} \
    --batch_size ${batch_size} \
    --gpu_ids ${gpu_ids} \
    --epoch ${epoch_checkpoints}
    "
    #--use_skinmask \
    
echo $CMD
eval $CMD

