_base_ = '../mae/mae_vit-base-p16_16xb256-coslr-400e-fp16_in1k.py'

# model settings
model = dict(
    type='EMAE',
    head=dict(
        type='EMAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        predictor=dict(
            type='NonLinearNeck',
            in_channels=768,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False),
        temperature=0.2,
        lamb=0.1))
