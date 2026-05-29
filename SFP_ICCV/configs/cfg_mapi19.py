_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_cityscapes_19.txt',
    prob_thd=0.15,
)

# dataset settings
dataset_type = 'MapillaryCityscapes19Dataset'
data_root = '/data/JS/OVDG/data/mapillary'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 560), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='validation/images',
            seg_map_path='validation/labels_TrainIds'),
        pipeline=test_pipeline))

