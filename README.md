# EfficientDet-MXNet
MXNet implementation of EfficientDet object detection as described in [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070) by Mingxing Tan, Ruoming Pang and Quoc V. Le.

## Prerequisites
* Python 3.6+
* MXNet 1.5.1+
* gluoncv 0.6.0

## Dataset
- run [mscoco.py](https://github.com/dmlc/gluon-cv/blob/master/scripts/datasets/mscoco.py)
 to download coco2017 dataset.
- run [pascal_voc.py](https://github.com/dmlc/gluon-cv/blob/master/scripts/datasets/pascal_voc.py) 
 to download voc dataset.

## Pretrained Model
- Will be provided

## Training EfficientDet
- COCO Dataset:
```
 sh train_efficientdet_coco.sh
```

## Testing EfficientDet
- COCO Dataset
```
python demo_efficientdet.py
```