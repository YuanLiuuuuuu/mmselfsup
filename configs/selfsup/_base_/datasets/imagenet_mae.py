# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmcls.ToPIL', to_rgb=True),
    dict(type='mmcls.torchvision/Resize', size=224),
    dict(
        type='mmcls.torchvision/RandomCrop',
        size=224,
        padding=4,
        padding_mode='reflect'),
    dict(type='mmcls.torchvision/RandomHorizontalFlip', p=0.5),
    dict(type='mmcls.ToNumpy', to_rgb=True),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
