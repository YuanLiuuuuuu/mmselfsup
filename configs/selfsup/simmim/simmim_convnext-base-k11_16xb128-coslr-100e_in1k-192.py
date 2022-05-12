_base_ = 'simmim_convnext-base_16xb128-coslr-100e_in1k-192.py'

# model settings
model = dict(backbone=dict(kernel_size=11))