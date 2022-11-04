_base_ = 'vit-large-p16_ft-8xb128-coslr-arcface-30e_in1k-448.py'

# model settings
model = dict(head=dict(loss=dict(type='mmselfsup.SoftmaxEQLLoss', num_classes=5000)))
