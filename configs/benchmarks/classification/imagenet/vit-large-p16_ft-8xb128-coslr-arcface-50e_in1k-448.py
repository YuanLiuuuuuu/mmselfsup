_base_ = 'vit-large-p16_ft-8xb128-coslr-50e_in1k-448.py'

# model settings
model = dict(
    head=dict(
        _delete_=True,
        type='mmselfsup.ArcFaceClsHeadAdaptiveMargin',
        num_classes=5000,
        in_channels=1024,
        number_sub_center=3,
        # ls_eps=0.1,
        ann_file='/mnt/cache/liuyuan/research/accv/filter_data/all_new.txt',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=None)

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.001))