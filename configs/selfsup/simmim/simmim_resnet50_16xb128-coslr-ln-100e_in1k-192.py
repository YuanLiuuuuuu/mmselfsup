_base_ = 'simmim_resnet50_16xb128-coslr-100e_in1k-192.py'

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='LN2d', eps=1e-6)))
