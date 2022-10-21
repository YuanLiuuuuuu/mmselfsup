_base_ = 'vit-large-p16_ft-8xb128-coslr-50e_in1k-448.py'

_base_ = 'vit-large-p16_ft-8xb128-coslr-50e_in1k.py'

# model settings
model = dict(backbone=dict(img_size=512))

# pipeline settings
# file_client_args = dict(backend='disk')
file_client_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/pymc/pcs_server_list.conf',
    client_cfg='/mnt/lustre/share/pymc/mc.conf',
    sys_path='/mnt/lustre/share/pymc',
)
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        scale=512,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=585,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=512),
    dict(type='PackClsInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.001))