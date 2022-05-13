_base_ = 'simmim_convnext-base_16xb128-coslr-100e_in1k-192.py'

# custom hook
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]
