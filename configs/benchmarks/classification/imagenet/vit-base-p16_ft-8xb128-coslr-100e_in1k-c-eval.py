_base_ = 'vit-base-p16_ft-8xb128-coslr-100e_in1k-eval.py'


test_evaluator = dict(type='CorruptionError', topk=(1, 5))