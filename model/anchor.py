#coding=utf-8
from __future__ import absolute_import
import mxnet as mx
import numpy as np


class AnchorGenerator(mx.gluon.HybridBlock):
    """
    Bounding box anchor generator for EfficientDet
    """
    def __init__(self, index, im_size, ratios, scales, step, alloc_size=(128, 128), 
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        super(AnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        self._num_anchors = len(ratios)*len(scales)
        anchors = self._generate_anchors(ratios, scales, step, alloc_size, offsets)
        self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)

    def _generate_anchors(self, ratios, scales, step, alloc_size, offsets):
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                for s in scales:
                    a_size = step*s
                    for r in ratios:
                        w = a_size*r[0]
                        h = a_size*r[1]
                        anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)    

    @property
    def num_anchors(self):
        return self._num_anchors

    def hybrid_forward(self, F, x, anchors):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = F.concat(*[cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)], dim=-1)
        return a.reshape((1, -1, 4))
