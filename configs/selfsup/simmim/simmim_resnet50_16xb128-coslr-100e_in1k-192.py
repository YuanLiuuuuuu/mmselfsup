_base_ = 'simmim_swin-base_16xb128-coslr-100e_in1k-192.py'

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SimMIMResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(in_channels=2048))