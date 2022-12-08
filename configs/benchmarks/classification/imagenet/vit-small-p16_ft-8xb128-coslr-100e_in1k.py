_base_ = 'vit-base-p16_ft-8xb128-coslr-100e_in1k.py'

# model settings
model = dict(backbone=dict(arch='deit-s'), head=dict(in_channels=384))
