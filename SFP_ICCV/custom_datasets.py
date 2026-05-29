import os.path as osp
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

CITYSCAPES_19_METAINFO = dict(
    classes=(
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'),
    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
             [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35],
             [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
             [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
             [0, 0, 230], [119, 11, 32]])

INPAINT_41_METAINFO = dict(
    classes=(
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle', 'bag', 'ball', 'barrel', 'bench', 'bird', 'bottle',
        'cart', 'cat', 'chair', 'cow', 'deer', 'dog', 'drone',
        'elephant', 'hat', 'horse', 'robot', 'sheep', 'table', 'toy',
        'umbrella', 'zebra'),
    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
             [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35],
             [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
             [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
             [0, 0, 230], [119, 11, 32], [242, 172, 36], [142, 84, 242],
             [29, 198, 42], [198, 69, 97], [36, 140, 242], [210, 242, 84],
             [183, 29, 198], [69, 198, 149], [242, 103, 36], [90, 84, 242],
             [73, 198, 29], [198, 69, 140], [36, 209, 242],
             [242, 221, 84], [127, 29, 198], [69, 198, 106],
             [242, 36, 38], [84, 132, 242], [129, 198, 29],
             [198, 69, 183], [36, 242, 206], [242, 168, 84]])

MAPILLARY_30_METAINFO = dict(
    classes=('road', 'sidewalk', 'building', 'wall', 'bridge', 'tunnel',
             'traffic sign', 'traffic light', 'pole', 'fence', 'sky',
             'vegetation', 'terrain', 'water', 'snow', 'sand', 'person',
             'rider', 'car', 'truck', 'bus', 'train', 'bicycle',
             'motorcycle', 'animal', 'signboard', 'railway', 'boat',
             'chair', 'trash can'),
    palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
             [102, 102, 156], [150, 100, 100], [150, 120, 90],
             [220, 220, 0], [250, 170, 30], [153, 153, 153],
             [190, 153, 153], [70, 130, 180], [107, 142, 35],
             [152, 251, 152], [0, 170, 255], [190, 255, 255],
             [245, 240, 170], [220, 20, 60], [255, 0, 0], [0, 0, 142],
             [0, 0, 70], [0, 60, 100], [0, 80, 100], [119, 11, 32],
             [0, 0, 230], [100, 170, 30], [255, 200, 0], [140, 150, 230],
             [0, 100, 180], [160, 80, 45], [90, 90, 90]])

ROADWORK_10_METAINFO = dict(
    classes=('road', 'sidewalk', 'barrier', 'police vehicle',
             'work vehicle', 'police officer', 'worker', 'cone',
             'arrow board', 'ttc sign'),
    palette=[[128, 64, 128], [244, 35, 232], [246, 116, 185],
             [255, 68, 51], [255, 104, 66], [184, 107, 35],
             [205, 135, 29], [30, 119, 179], [241, 71, 14], [254, 139, 32]])

@DATASETS.register_module()
class PascalVOC20Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[128, 0, 0], [0, 128, 0], [0, 0, 192],
                 [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 128, 64],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)

@DATASETS.register_module()
class COCOObjectDataset(BaseSegDataset):
    """
    Implementation borrowed from TCL (https://github.com/kakaobrain/tcl) and GroupViT (https://github.com/NVlabs/GroupViT)
    COCO-Object dataset.
    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    METAINFO = dict(
    
    classes = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),

    palette = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
               [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
               [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
               [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
               [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
               [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
               [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
               [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
               [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0],
               [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]])

    def __init__(self, **kwargs):
        super(COCOObjectDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)

@DATASETS.register_module()
class PascalContext60Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bag', 'bed', 'bedclothes',
                 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog',
                 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                 'horse', 'keyboard', 'light', 'motorbike', 'mountain',
                 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road',
                 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow',
                 'sofa', 'table', 'track', 'train', 'tree', 'truck',
                 'tvmonitor', 'wall', 'water', 'window', 'wood'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class PascalContext59Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
                 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
                 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
                 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
                 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
                 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
                 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
                 'wall', 'water', 'window', 'wood'),
        palette=[[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                 [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class Cityscapes19LikeDataset(BaseSegDataset):
    """Cityscapes-19 style dataset for cross-domain evaluation."""

    METAINFO = CITYSCAPES_19_METAINFO

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class ACDCCityscapes19Dataset(Cityscapes19LikeDataset):
    """ACDC dataset prepared in s2_corr layout."""

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)


@DATASETS.register_module()
class BDD100KCityscapes19Dataset(Cityscapes19LikeDataset):
    """BDD100K semantic segmentation dataset in train-id format."""

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_train_id.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)


@DATASETS.register_module()
class Inpaint41Dataset(BaseSegDataset):
    """41-class inpaint dataset used by ACDC/BDD open-vocabulary evaluation."""

    METAINFO = INPAINT_41_METAINFO

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_gt.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class ACDCInpaint41Dataset(Inpaint41Dataset):
    """ACDC inpaint-41 dataset."""


@DATASETS.register_module()
class BDD100KInpaint41Dataset(Inpaint41Dataset):
    """BDD100K inpaint-41 dataset."""


@DATASETS.register_module()
class MapillaryCityscapes19Dataset(Cityscapes19LikeDataset):
    """Mapillary dataset remapped to Cityscapes-19 labels."""

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)


@DATASETS.register_module()
class Mapillary30Dataset(BaseSegDataset):
    """Mapillary dataset remapped to 30 open-vocabulary classes."""

    METAINFO = MAPILLARY_30_METAINFO

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class Roadwork10Dataset(BaseSegDataset):
    """ROADWork dataset with recursive image/label matching."""

    METAINFO = ROADWORK_10_METAINFO
    IMG_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_trainIds.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)

    def load_data_list(self):
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        suffix_len = len(self.seg_map_suffix)

        for seg_map in fileio.list_dir_or_file(
                dir_path=ann_dir,
                list_dir=False,
                suffix=self.seg_map_suffix,
                recursive=True,
                backend_args=self.backend_args):
            rel_stem = seg_map[:-suffix_len]
            img_path = None
            for img_suffix in self.IMG_SUFFIXES:
                candidate = osp.join(img_dir, rel_stem + img_suffix)
                if fileio.exists(candidate, backend_args=self.backend_args):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            data_info = dict(
                img_path=img_path,
                seg_map_path=osp.join(ann_dir, seg_map),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[])
            data_list.append(data_info)

        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list