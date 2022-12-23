_base_ = 'mae-plus_vit-base-p16_8xb512-amp-coslr-300e_in1k-ms.py'

# model settings
model = dict(backbone=dict(out_indices=[3, 11]))