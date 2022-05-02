model = dict(
    type='Classification',
    backbone=dict(
        type='MIMVisionTransformer',
        arch='b',
        patch_size=16,
        final_norm=True,
        finetune=False,
        out_indices=-1),
    head=dict(type='MAELinprobeHead', num_classes=1000, embed_dim=768))
