_base_ = 'vit-large-p16_linear-8xb2048-coslr-90e_in1k.py'

# dataloaders
# Use PyTorch Transform
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
    dict(type='mmselfsup.PytorchRandomResizedCrop', size=224, interpolation=3),
    dict(type='mmselfsup.PytorchRandomHorizontalFlip', p=0.5),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmselfsup.PytorchResize', size=256, interpolation=3),
    dict(type='mmselfsup.PytorchCenterCrop', size=224),
    dict(type='PackClsInputs'),
]
train_dataloader = dict(
    batch_size=2048, dataset=dict(pipeline=train_pipeline), drop_last=True)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline), drop_last=False)
test_dataloader = dict(dataset=dict(pipeline=test_pipeline), drop_last=False)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        by_epoch=True,
        begin=10,
        end=50,
        eta_min=0.0,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=50)