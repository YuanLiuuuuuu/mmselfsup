_base_ = 'swin-base_ft-8xb256-coslr-100e_in1k.py'

# dataset
data = dict(samples_per_gpu=128)
