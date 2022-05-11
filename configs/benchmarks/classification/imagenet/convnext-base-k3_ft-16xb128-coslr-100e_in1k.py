_base_ = 'convnext-base_ft-16xb128-coslr-100e_in1k.py'

# model settings
model = dict(backbone=dict(kernel_size=3))
