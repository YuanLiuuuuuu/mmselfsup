_base_ = ['swin-base_ft-8xb256-coslr-100e_in1k.py']

# model settings
model = dict(
    backbone=dict(
        type='ConvNeXt',
        arch='base',
        out_indices=(3, ),
        drop_path_rate=0.5,
        gap_before_final_norm=True,
        _delete_=True),
    neck=None)

# optimizer settings
custom_imports = dict(imports='mmselfsup.engine', allow_failed_imports=False)
optim_wrapper = dict(
    optimizer=dict(model_type='convnext', layer_decay_rate=0.9))