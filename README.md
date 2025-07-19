# [Deep semantic center-guided hashing for multi-label cross-modal retrieval](https://www.sciencedirect.com/science/article/pii/S0957417425023656)


## Training

### Processing dataset and CLIP pretrained model
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Start

> python main_cls.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --clip-path ./ViT-B-32.pt --batch-size 256 
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 256

## Citation
``` 
@article{SUI2026128747,
title = {Deep semantic center-guided hashing for multi-label cross-modal retrieval},
journal = {Expert Systems with Applications},
volume = {295},
pages = {128747},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.128747},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425023656},
author = {Xinzheng Sui and Lei Wu and Yadong Huo and Qibing Qin and Lei Huang and Wenfeng Zhang},
}
```
