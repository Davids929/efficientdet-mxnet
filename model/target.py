#coding=utf-8
"""EfficientDet training target generator."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon
from gluoncv.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from gluoncv.nn.sampler import OHEMSampler, NaiveSampler
from gluoncv.nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from gluoncv.nn.bbox import BBoxCenterToCorner

class TargetGenerator(gluon.Block):
    """Training targets generator for EfficientDet.

    Parameters
    ----------
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    neg_thresh : float
        IOU overlap threshold for negative mining, default is 0.5.
    negative_mining_ratio : float
        Ratio of hard vs positive for negative mining.
    stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """
    def __init__(self, iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(TargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher(
            [BipartiteMatcher(share_max=False), MaximumMatcher(iou_thresh)])
        if negative_mining_ratio > 0:
            self._sampler = OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
            self._use_negative_sampling = True
        else:
            self._sampler = NaiveSampler()
            self._use_negative_sampling = False
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    def forward(self, anchors, cls_preds, gt_boxes, gt_ids):
        """Generate training targets."""
        
        anchors = self._center_to_corner(anchors.reshape((-1, 4)))
        ious = mx.nd.transpose(mx.nd.contrib.box_iou(anchors, gt_boxes), (1, 0, 2))
        matches = self._matcher(ious)
        if self._use_negative_sampling:
            samples = self._sampler(matches, cls_preds, ious)
        else:
            samples = self._sampler(matches)
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        return cls_targets, box_targets, box_masks