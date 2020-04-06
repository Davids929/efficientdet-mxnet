# EfficientDet-MXNet
MXNet implementation of EfficientDet object detection as described in [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070) by Mingxing Tan, Ruoming Pang and Quoc V. Le.

## prerequisites
* Python 3.6+
* MXNet 1.5.1+
* gluoncv 0.6.0

## Dataset
run [COCO2017.sh](https://github.com/toandaominh1997/EfficientDet.Pytorch/blob/master/datasets/scripts/COCO2017.sh)

## Training EfficientDet
- COCO Dataset:
```
 sh train_efficientdet_coco.sh
```