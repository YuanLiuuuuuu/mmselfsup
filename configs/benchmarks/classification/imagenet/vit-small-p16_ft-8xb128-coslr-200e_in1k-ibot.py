_base_ = 'vit-small-p16_ft-8xb128-coslr-100e_in1k.py'

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-3,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
        model_type='vit',  # layer-wise lr decay type
        layer_decay_rate=0.75))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=195,
        by_epoch=True,
        begin=5,
        end=200,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=200)