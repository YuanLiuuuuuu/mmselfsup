# dataset settings
data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    test_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
file_client_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/memcached_client/pcs_server_list.conf',
    client_cfg='/mnt/lustre/share_data/zhangwenwei/software/pymc/mc.conf',
    sys_path='/mnt/lustre/share_data/zhangwenwei/software/pymc')
data = dict(
    samples_per_gpu=32,  # total 32x8=256, 8GPU linear cls
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
            file_client_args=file_client_args),
        pipeline=train_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/val',
            ann_file='data/imagenet/meta/val.txt',
            file_client_args=file_client_args),
        pipeline=test_pipeline,
        prefetch=prefetch))
evaluation = dict(interval=10, topk=(1, 5))
