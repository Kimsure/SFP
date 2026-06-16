_base_ = './cfg_acdc19_orig.py'

# ACDC snow condition, val split, original-name Cityscapes-19 vocabulary.
test_dataloader = dict(
    dataset=dict(
        seg_map_suffix='_gt_labelTrainIds.png',
        data_prefix=dict(
            img_path='rgb_anon/snow/val',
            seg_map_path='gt/snow/val')))
