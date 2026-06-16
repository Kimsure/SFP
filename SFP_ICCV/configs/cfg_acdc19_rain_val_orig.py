_base_ = './cfg_acdc19_orig.py'

# ACDC rain condition, val split, original-name Cityscapes-19 vocabulary.
test_dataloader = dict(
    dataset=dict(
        seg_map_suffix='_gt_labelTrainIds.png',
        data_prefix=dict(
            img_path='rgb_anon/rain/val',
            seg_map_path='gt/rain/val')))
