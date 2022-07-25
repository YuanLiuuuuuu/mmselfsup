_base_ = ['swin-base_ft-8xb256-coslr-100e_in1k.py']

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SimMIMResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        mask=False),
    neck=dict(in_channels=2048))

# optimizer settings
custom_imports = dict(imports='mmselfsup.engine', allow_failed_imports=False)
optim_wrapper = dict(
    optimizer=dict(model_type='convnext', layer_decay_rate=0.9))