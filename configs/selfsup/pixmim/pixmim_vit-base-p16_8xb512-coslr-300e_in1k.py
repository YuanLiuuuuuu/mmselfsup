_base_ = 'pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0024, betas=(0.9, 0.95), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            ln=dict(decay_mult=0.0),
            bias=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            mask_token=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0))),
    _delete_=True)