# EfficientDet-MXNet
MXNet implementation of EfficientDet object detection as described in [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070) by Mingxing Tan, Ruoming Pang and Quoc V. Le.

## prerequisites
* Python 3.6+
* MXNet 1.5.1+
* gluoncv 0.6.0

## Dataset
- download coco2017 
  python [mscoco.py](https://github.com/dmlc/gluon-cv/blob/master/scripts/datasets/mscoco.py)
- download voc
  python [pascal_voc.py](https://github.com/dmlc/gluon-cv/blob/master/scripts/datasets/pascal_voc.py)
## Training EfficientDet
- COCO Dataset:
```
 sh train_efficientdet_coco.sh
```