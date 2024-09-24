# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train Phase 1 Mask2Map with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/mask2map/M2M_nusc_r50_full_1Phase_12n12ep.py 8
```

Train Phase 2 Mask2Map with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/mask2map/M2M_nusc_r50_full_2Phase_12n12ep.py 8
```

Eval Mask2Map with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/mask2map/M2M_nusc_r50_full_2Phase_12n12ep.py ./path/to/ckpts.pth 8
```