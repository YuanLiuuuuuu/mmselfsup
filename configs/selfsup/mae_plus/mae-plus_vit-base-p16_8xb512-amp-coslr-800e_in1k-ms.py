_base_ = 'mae-plus_vit-base-p16_8xb512-amp-coslr-300e_in1k-ms.py'

# pre-train for 1600 epochs
train_cfg = dict(max_epochs=800)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=760,
        by_epoch=True,
        begin=40,
        end=800,
        convert_to_iter_based=True)
]