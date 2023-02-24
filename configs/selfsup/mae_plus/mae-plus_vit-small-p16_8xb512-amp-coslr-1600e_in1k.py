_base_ = 'mae-plus_vit-small-p16_8xb512-amp-coslr-300e_in1k.py'

# pre-train for 100 epochs
train_cfg = dict(max_epochs=1600)