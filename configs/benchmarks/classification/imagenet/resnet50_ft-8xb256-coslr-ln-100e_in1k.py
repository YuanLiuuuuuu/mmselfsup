_base_ = 'resnet50_ft-8xb256-coslr-100e_in1k.py'

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='LN2d', eps=1e-6)))