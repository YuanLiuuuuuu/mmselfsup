_base_ = 'mae-plus_vit-base-p16_8xb512-amp-coslr-1600e_in1k.py'

# model settings
model = dict(backbone=dict(out_indices=[3, 5, 7, 11]))