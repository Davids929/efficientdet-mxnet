#coding=utf-8

import mxnet as mx
from mxnet import gluon, nd
from gluoncv.loss import FocalLoss
# from mxnet.gluon.loss import HuberLoss

def _as_list(arr):
    """Make sure input is a list of mxnet NDArray"""
    if not isinstance(arr, (list, tuple)):
        return [arr]
    return arr

class EfficientDetLoss(gluon.Block):
    def __init__(self, num_classes, negative_mining_ratio=3, rho=1.0, lambd=1.0,
                 min_hard_negatives=0, **kwargs):
        super(EfficientDetLoss, self).__init__(**kwargs)
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd
        self._min_hard_negatives = max(0, min_hard_negatives)
        self.focal_loss = FocalLoss(size_average=False, gamma=1.5, num_class=num_classes)

    def forward(self, cls_pred, box_pred, cls_target, box_target):
        
        cls_pred, box_pred, cls_target, box_target = [_as_list(x) \
            for x in (cls_pred, box_pred, cls_target, box_target)]

        # cross device reduction to obtain positive samples in entire batch
        num_pos = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pos_samples = (ct > 0)
            num_pos.append(pos_samples.sum())
        num_pos_all = sum([p.asscalar() for p in num_pos])
        if num_pos_all < 1 and self._min_hard_negatives < 1:
            # no positive samples and no hard negatives, return dummy losses
            cls_losses = [nd.sum(cp * 0) for cp in cls_pred]
            box_losses = [nd.sum(bp * 0) for bp in box_pred]
            sum_losses = [nd.sum(cp * 0) + nd.sum(bp * 0) for cp, bp in zip(cls_pred, box_pred)]
            return sum_losses, cls_losses, box_losses
        
        cls_losses = []
        box_losses = []
        sum_losses = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pos = ct > 0
            cls_loss = self.focal_loss(cp, ct)
            cls_losses.append(cls_loss/ max(1., num_pos_all))

            bp = nd.reshape_like(bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / max(1., num_pos_all))
            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])

        return sum_losses,cls_losses, box_losses