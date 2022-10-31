_base_ = 'vit-large-p16_ft-8xb128-coslr-arcface-50e_in1k-448.py'

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=25,
        by_epoch=True,
        begin=5,
        end=30,
        eta_min=1e-06,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)