## Feature Purificatrion Matters: Supressing outlier propagation for training-free open-vocabulary semantic segmentation, ICCV25

### Abstract

Read Paper [HERE](https://openaccess.thecvf.com/content/ICCV2025/papers/Jin_Feature_Purification_Matters_Suppressing_Outlier_Propagation_for_Training-Free_Open-Vocabulary_Semantic_ICCV_2025_paper.pdf).

---
Training-free open-vocabulary semantic segmentation has advanced with vision-language models like CLIP, which exhibit strong zero-shot abilities. However, CLIP’s attention mechanism often wrongly emphasises specific image tokens, namely outliers, which results in irrelevant over-activation. Existing approaches struggle with these outliers that arise in intermediate layers and propagate through the model, ultimately degrading spatial perception. In this paper, we propose a Self-adaptive Feature Purifier framework (SFP) to suppress propagated outliers and enhance semantic representations for open-vocabulary semantic segmentation. Specifically, based on an in-depth analysis of attention responses between image and class tokens, we design a selfadaptive outlier mitigator to detect and mitigate outliers at each layer for propagated feature purification. In addition, we introduce a semantic-aware attention enhancer to augment attention intensity in semantically relevant regions, which strengthens the purified feature to focus on objects. Further, we introduce a hierarchical attention integrator to aggregate multi-layer attention maps to refine spatially coherent feature representations for final segmentation. Our proposed SFP enables robust outlier suppression and object-centric feature representation, leading to a more precise segmentation. Extensive experiments show that our method achieves state-of-the-art performance and surpasses existing methods by an average of 4.6% mIoU on eight segmentation benchmarks.

---

### Visualization of Outlier Propagation

You can check and verify the attention outlier at [Notebook](outlier_vis.ipynb).

[fig](./figs/outlier_vis.png)


### Installation
---
```
git clone https://github.com/Kimsure/SFP.git
cd SFP
```

#### 1) Recommended

Please use the following commands to install the same conda environment.
```
conda env create f environment.yml
```

#### 2) Manual Setup

If the above YAML file can't work properly, you can manually install the following required packages.

```
conda create -n sfp_ovss python=3.9
conda activate sfp_ovss
pip install torch==2.0.0 torchvision==0.15.1
pip install scikit-learn scikit-image
pip install mim
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install ftfy regex numpy==1.26.4 yapf==0.40.1
```

### Datasets
---
Plase follow SCLIP and mmsegentation_document.

### Quick Start
---
We currently provide a visualization notebook to evaluate the outler detector. You can check and verify it at [HERE](outlier_vis.ipynb).

### TODOs
---
We are still organizing our codes and will release the remaining parts once ready. Please stay tuned.

### Citations
---
If you find our work useful, please cite this paper:
```
@inproceedings{jin2025feature,
  title={Feature Purification Matters: Suppressing Outlier Propagation for Training-Free Open-Vocabulary Semantic Segmentation},
  author={Jin, Shuo and Yu, Siyue and Zhang, Bingfeng and Sun, Mingjie and Dong, Yi and Xiao, Jimin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

### Acknowledgement
---
This repository was developed based on [CLIPtrase](https://github.com/leaves162/CLIPtrase), [SCLIP](https://github.com/wangf3014/SCLIP), etc. Thanks for their great works!