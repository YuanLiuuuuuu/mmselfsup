_base_ = 'mae-plus_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

# model settings
model = dict(backbone=dict(mask_ratio=0.8))
