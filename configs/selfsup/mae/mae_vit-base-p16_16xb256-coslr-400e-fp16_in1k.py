_base_ = 'mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k.py'

# dataset
data = dict(samples_per_gpu=256, workers_per_gpu=16)