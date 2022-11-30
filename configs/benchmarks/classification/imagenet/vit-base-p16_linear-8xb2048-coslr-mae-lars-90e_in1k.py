_base_ = 'vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'

# optimizer
optimizer = dict(type='mmselfsup.MAE_LARS', lr=6.4, weight_decay=0.0)
optim_wrapper = dict(
    type='AmpOptimWrapper', optimizer=optimizer, _delete_=True)
