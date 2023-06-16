_base_ = 'faster_rcnn_r50_fpn_straug_3x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3)))

data = dict(
    train=dict(type='RobustODAV',
        ann_file='data/saod/annotations/saod_av_train.json',
        img_prefix='data/nuimages/'),
    val=dict(type='RobustODAV',
        ann_file='data/saod/annotations/saod_av_val.json',
        img_prefix='data/nuimages/'),
    test=dict(type='RobustODAV',
        ann_file='data/saod/annotations/obj45k.json',
        img_prefix='data/'))