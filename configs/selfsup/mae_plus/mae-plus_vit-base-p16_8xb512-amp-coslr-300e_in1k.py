_base_ = '../mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

# dataloader settings
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/imagenet':
        's3://openmmlab/datasets/classification/imagenet',
        'data/imagenet':
        's3://openmmlab/datasets/classification/imagenet'
    }))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmselfsup.PytorchSimpleResizedCrop', size=224),
    dict(type='mmselfsup.PytorchRandomHorizontalFlip', p=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# model settings
model = dict(
    type='MAEPlus',
    target_generator=dict(
        type='LowFreqTargetGenerator', radius=40, img_size=224),
)

# randomness
randomness = dict(seed=2, diff_rank_seed=True)
