_base_ = 'mae-plus_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

# model settings
model = dict(
    backbone=dict(arch='deit-s'),
    neck=dict(embed_dim=384, decoder_embed_dim=384, decoder_num_heads=12))
