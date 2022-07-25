from requests import head

_base_ = ['swin-base_ft-8xb256-coslr-100e_in1k.py']

# model settings
custom_imports = dict(
    imports=['mmselfsup.engine', 'mmselfsup.models'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmselfsup.SimMIMResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        mask=False),
    head=dict(in_channels=2048))

# optimizer settings
optim_wrapper = dict(optimizer=dict(model_type='resnet', layer_decay_rate=0.9))
