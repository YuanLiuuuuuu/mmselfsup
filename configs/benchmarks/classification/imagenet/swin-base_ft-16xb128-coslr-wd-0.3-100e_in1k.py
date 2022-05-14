_base_ = 'swin-base_ft-16xb128-coslr-100e_in1k.py'

# optimizer
optimizer = dict(weight_decay=0.3)
