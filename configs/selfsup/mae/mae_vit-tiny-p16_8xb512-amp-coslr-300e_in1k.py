_base_ = 'mae_vit-small-p16_8xb512-amp-coslr-300e_in1k.py'

# model settings
model = dict(
    backbone=dict(arch='deit-t', out_indices=-1),
    neck=dict(embed_dim=192, decoder_embed_dim=192, decoder_num_heads=12))
