_base_ = 'mae-plus_vit-base-p16_8xb512-amp-coslr-300e_in1k-0.8m-ms.py'

# model settings
model = dict(backbone=dict(out_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))