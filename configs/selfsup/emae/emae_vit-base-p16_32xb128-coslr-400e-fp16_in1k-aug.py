_base_ = 'emae_vit-base-p16_16xb256-coslr-400e-fp16_in1k-aug.py'

data = dict(samples_per_gpu=128, workers_per_gpu=16)