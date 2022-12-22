_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = '/home/beanbagthomas/code/hrf/mmdetection/checkpoints/swin_large_patch4_window7_224_22kto1k.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        embed_dims=192,
        drop_path_rate=0.2,
        num_heads=[6, 12, 24, 48],
        window_size=7,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    roi_head=dict(
        bbox_head=dict(num_classes=7),
        mask_head=dict(num_classes=7)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-05,
    betas=(0.9, 0.999),
    weight_decay=1e-8,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
"""
MODEL:

TRAIN:
  EPOCHS: 30
  WARMUP_EPOCHS: 5


  WARMUP_LR: 2e-08
  MIN_LR: 2e-07
  """