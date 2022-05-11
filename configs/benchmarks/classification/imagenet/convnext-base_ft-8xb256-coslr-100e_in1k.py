_base_ = 'convnext-base_ft-16xb128-coslr-100e_in1k.py'

data = dict(samples_per_gpu=256, workers_per_gpu=32)
