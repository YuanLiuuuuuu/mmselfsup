_base_ = [
    '../_base_/models/vit-base-p16_linprobe.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]

# dataset
file_client_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/memcached_client/pcs_server_list.conf',
    client_cfg='/mnt/lustre/share_data/zhangwenwei/software/pymc/mc.conf',
    sys_path='/mnt/lustre/share_data/zhangwenwei/software/pymc',
)
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
data = dict(
    imgs_per_gpu=512,
    drop_last=False,
    workers_per_gpu=32,
    train=dict(
        pipeline=train_pipeline,
        data_source=dict(file_client_args=file_client_args)),
    val=dict(
        pipeline=test_pipeline,
        data_source=dict(file_client_args=file_client_args)))

# optimizer
optimizer = dict(lr=0.1 * 4096 / 256, weight_decay=0.0)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=1, out_dir='')
persistent_workers = True
runner = dict(type='EpochBasedRunner', max_epochs=90)
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])