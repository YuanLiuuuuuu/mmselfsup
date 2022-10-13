_base_ = '../mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

# model settings
# Compared to MAE, backbone and neck are changed. Please refer to the
# implementation of these blocks, and fill in the corresponding params here.
model = dict(
    backbone=dict(type='ConvMAEViT'), neck=dict(type='ConvMAEPretrainDecoder'))

# pre-train for 100 epochs, you can extend it to longer epochs
train_cfg = dict(max_epochs=100)

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
        T_max=60,
        by_epoch=True,
        begin=40,
        end=100,
        convert_to_iter_based=True)
]

# optimizer wrapper
optim_wrapper = dict(optimizer=dict(lr=1.5e-4 * 1024 / 256))
