_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_roadwork_10_original_name.txt',
    prob_thd=0.15,
)

# dataset settings
dataset_type = 'Roadwork10Dataset'
data_root = '/data/JS/OVDG/data/ROADWork_Data'

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
            img_path='images',
            seg_map_path='gtFine_10/val'),
        pipeline=test_pipeline))

