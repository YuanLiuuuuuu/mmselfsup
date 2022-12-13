_base_ = 'vit-base-p16_linear-8xb2048-coslr-mae-lars-torch-90e_in1k.py'

# dataloaders
# Use PyTorch Transform
file_client_args = dict(backend='backend')
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