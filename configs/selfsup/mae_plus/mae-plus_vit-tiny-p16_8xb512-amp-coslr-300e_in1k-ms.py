_base_ = 'mae-plus_vit-small-p16_8xb512-amp-coslr-300e_in1k-ms.py'

# model settings
model = dict(
    backbone=dict(arch='deit-s', out_indices=[3, 5, 7, 11]),
    neck=dict(embed_dim=192, decoder_embed_dim=192, decoder_num_heads=12))