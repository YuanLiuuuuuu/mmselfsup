_base_ = 'simmim_swin-base_16xb128-coslr-100e_in1k-192.py'

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SimMIMConvNext',
        arch='base',
        out_indices=(3, ),
        drop_path_rate=0.5,
        gap_before_final_norm=False,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]))