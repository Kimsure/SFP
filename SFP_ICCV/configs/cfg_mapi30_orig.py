_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_mapi_30_original_name.txt',
    prob_thd=0.15,
)

# dataset settings
dataset_type = 'Mapillary30Dataset'
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
            img_path='OV_30/images',
            seg_map_path='OV_30/labels'),
        pipeline=test_pipeline))

