_base_ = 'vit-base-p16_linear-8xb2048-coslr-mae-lars-torch-90e_in1k.py'

# model settings
model = dict(
    backbone=dict(arch='deit-s'),
    neck=dict(input_features=384),
    head=dict(in_channels=384))
