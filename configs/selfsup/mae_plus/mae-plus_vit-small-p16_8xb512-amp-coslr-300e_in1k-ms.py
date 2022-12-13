_base_ = 'mae-plus_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

# model settings
model = dict(
    type='MAEPlus',
    backbone=dict(arch='deit-s', out_indices=[3, 5, 7, 11]),
    neck=dict(embed_dim=384, decoder_embed_dim=384, decoder_num_heads=12),
    target_generator=dict(
        type='LowFreqTargetGenerator', radius=40, img_size=224),
)
