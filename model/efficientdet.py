#coding=utf-8
import mxnet as mx
import os
from mxnet.gluon import nn
from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from .efficientnet import _add_conv, Activation, get_efficientnet
from .anchor import AnchorGenerator


bifpn_nodes_config = [
      {'step': 64,  'inputs_offsets': [3, 4]},
      {'step': 32,  'inputs_offsets': [2, 5]},
      {'step': 16,  'inputs_offsets': [1, 6]},
      {'step': 8,   'inputs_offsets': [0, 7]},
      {'step': 16,  'inputs_offsets': [1, 7, 8]},
      {'step': 32,  'inputs_offsets': [2, 6, 9]},
      {'step': 64,  'inputs_offsets': [3, 5, 10]},
      {'step': 128, 'inputs_offsets': [4, 11]},
  ]

class SeparableConvBlock(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, act_type='swish', 
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(SeparableConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.depthwise_conv = nn.Conv2D(in_channels, 3, strides=1, padding=1, groups=in_channels, use_bias=False)
            self.pointwise_conv = nn.Conv2D(out_channels, 1, strides=1)
            if norm_layer is not None:
                self.norm = norm_layer(scale=True, **norm_kwargs)
            else: 
                self.norm = None
            if act_type != None:
                self.act = Activation(act_type)
            else:
                self.act = None

    def hybrid_forward(self, F, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act != None:
            x = self.act(x)
        return x

class BiFPN(nn.HybridBlock):
    def __init__(self, channels, num_features=5, act_type='swish', weight_method='fastattn', 
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(BiFPN, self).__init__(**kwargs)
        self.num_features = num_features
        self.weight_method = weight_method
        assert len(bifpn_nodes_config) == num_features*2 - 2
        with self.name_scope():
            self.convs = nn.HybridSequential()
            self.weights = []
            for i in range(len(bifpn_nodes_config)):
                self.convs.add(SeparableConvBlock(channels, channels, act_type=act_type,
                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                weight_size = len(bifpn_nodes_config[i]['inputs_offsets'])
                weight_name = 'weights_%d'%i
                att_weight = self.params.get(weight_name, shape=(weight_size), init=mx.init.One())
                setattr(self, weight_name, att_weight)
                
    def weight_sum(self, F, features, weight, types='attn'):
        features = F.stack(*features, axis=-1)
        if types=='attn':
            normlize_weight = F.softmax(weight)
            features = F.dot(features, normlize_weight)
        elif types=='fastattn':
            edge_weight = F.relu(weight)
            sum_weight  = F.sum(edge_weight) + 0.0001
            edge_weight = F.broadcast_div(edge_weight, sum_weight)
            features = F.dot(features, edge_weight)
        else:
            features = mx.nd.sum(features, axis=-1)
        return features

    def hybrid_forward(self, F, *features, **weights):
        feats = list(features)
        for idx, block in enumerate(self.convs):
            input_ids = bifpn_nodes_config[idx]['inputs_offsets']
            weight = weights['weights_%d'%idx]
            inputs = []
            for i, idx in enumerate(input_ids):
                if i==0:
                    inputs.append(feats[idx])
                else:
                    feat = F.contrib.BilinearResize2D(feats[idx], like=inputs[0], mode='like')
                    inputs.append(feat)

            inputs = self.weight_sum(F, inputs, weight, types=self.weight_method)
            out = block(inputs)
            feats.append(out)
            output = feats[-self.num_features:]
        return output
        
class OutputSubnet(nn.HybridBlock):
    def __init__(self, channels, num_layers, out_channels, num_anchors, 
                 act_type='swish', norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(OutputSubnet, self).__init__(**kwargs)
        
        with self.name_scope():
            self.body = nn.HybridSequential()
            for i in range(num_layers):
                self.body.add(SeparableConvBlock(channels, channels, act_type=act_type,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            output_channels = out_channels*num_anchors
            self.output =SeparableConvBlock(channels, output_channels, norm_layer=None, act_type=None)

    def hybrid_forward(self, F, x):
        x = self.body(x)
        x = self.output(x)
        return x

class EfficientDet(nn.HybridBlock):
    """EfficientDet: https://arxiv.org/pdf/1911.09070.

    Parameters
    ----------
    base_size : int
        Base input size.
    stages : list of mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a backbone network with multi-output.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of all categories.
     
    fpn_channel : list of int
        Number of channels for BiFPN layers.
    fpn_repea : Number of layers for BiFPN.
    box_cls_repeat : Number of layers for box and class.

    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    anchor_alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `anchor_alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, scalar, etc.
    ctx : mx.Context
        Network context.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
        This will only apply to base networks that has `norm_layer` specified, will ignore if the
        base network (e.g. VGG) don't accept this argument.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    """

    def __init__(self, base_size, stages, ratios, scales, steps, classes, 
                 fpn_channel=64, fpn_repeat=3, box_cls_repeat=3, act_type='swish', 
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400,
                 post_nms=100, anchor_alloc_size=128, ctx=mx.cpu(),
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):

        super(EfficientDet, self).__init__(**kwargs)

        self.num_stages = len(steps)
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        num_anchors = len(ratios)*len(scales)
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
    
        im_size = (base_size, base_size)
        asz = anchor_alloc_size
        with self.name_scope():
            self.stages     = nn.HybridSequential()
            self.proj_convs = nn.HybridSequential()
            self.fpns       = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            for stage in stages:
                self.stages.add(stage)
            for i in range(self.num_stages):
                block = nn.HybridSequential()
                _add_conv(block, channels=fpn_channel, act_type=act_type, 
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                self.proj_convs.add(block)
                anchor_generator = AnchorGenerator(i, im_size, ratios, scales, steps[i], (asz, asz))
                self.anchor_generators.add(anchor_generator)
                asz = max(asz//2, 16)

            for i in range(fpn_repeat):
                self.fpns.add(BiFPN(fpn_channel, num_features=self.num_stages, act_type=act_type, 
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.cls_net = OutputSubnet(fpn_channel, box_cls_repeat, self.num_classes+1, num_anchors, act_type=act_type,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='class_net')
            self.box_net = OutputSubnet(fpn_channel, box_cls_repeat, 4, num_anchors, act_type=act_type,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='box_net')
            self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = MultiPerClassDecoder(self.num_classes+1, thresh=0.01)

    @property
    def num_classes(self):
        """Number of (non-background) categories.
        Returns
        -------
        int
            Number of (non-background) categories.
        """
        return len(self.classes)

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def hybrid_forward(self, F, x):
       
        feats = []
        # backbone forward
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        # additional stages
        for i in range(self.num_stages-len(feats)):
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
            feats.append(x)
        # The channel of feature project to the input channel of BiFPN 
        for i, block in enumerate(self.proj_convs):
            feats[i] = block(feats[i])
        # Binfpn forward
        for block in self.fpns:
            feats = block(*feats)
        
        cls_preds = []
        box_preds = []
        anchors   = []
        for feat, ag in zip(feats, self.anchor_generators):
            box_pred = self.box_net(feat)
            cls_pred = self.cls_net(feat)
            anchor   = ag(feat)
            # (b, c*a, h, w) -> (b, c, a*h*w)
            box_pred = F.reshape(F.transpose(box_pred, axes=(0, 2, 3, 1)), shape=(0, -1, 4))
            cls_pred = F.reshape(F.transpose(cls_pred, axes=(0, 2, 3, 1)), 
                                 shape=(0, -1, self.num_classes+1))
            cls_preds.append(cls_pred)
            box_preds.append(box_pred)
            anchors.append(anchor)
        
        cls_preds = F.concat(*cls_preds, dim=1)
        box_preds = F.concat(*box_preds, dim=1)
        anchors   = F.concat(*anchors, dim=1)
        if mx.autograd.is_training():
            return [cls_preds, box_preds, anchors]
        
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score  = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes
        

def efficientdet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:  backbone, input_size, fpn_channel, fpn_repeat, box_cls_repeat, anchor_scales
        'efficientdet-b0': ['efficientnet-b0', 512,  64,  3, 3, 4.0],
        'efficientdet-b1': ['efficientnet-b1', 640,  88,  4, 3, 4.0],
        'efficientdet-b2': ['efficientnet-b2', 768,  112, 5, 3, 4.0],
        'efficientdet-b3': ['efficientnet-b3', 896,  160, 5, 3, 4.0],
        'efficientdet-b4': ['efficientnet-b4', 1024, 224, 7, 4, 4.0],
        'efficientdet-b5': ['efficientnet-b5', 1280, 288, 7, 4, 4.0],
        'efficientdet-b6': ['efficientnet-b6', 1280, 384, 8, 5, 4.0],
        'efficientdet-b7': ['efficientnet-b7', 1536, 384, 8, 5, 5.0]
    }
    if model_name not in list(params_dict.keys()):
        raise NotImplementedError('%s is not exists.'%model_name)

    return params_dict[model_name]

def get_efficientdet(model_name, classes, 
                     pretrained=False, pretrained_base=False, ctx=mx.cpu(),
                     root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    
    model_config = efficientdet_params(model_name)
    backbone_name, base_size, fpn_c, num_fpn, box_cls_repeat, anchor_scales = model_config
    ratios  = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    scales  = [anchor_scales*i for i in [2**0, 2**(1/3), 2**(2/3)]]
    steps   = [8, 16, 32, 64, 128]
    
    base_net     = get_efficientnet(backbone_name)
    if pretrained_base:
        base_net.load_parameters(os.path.join(root, backbone_name + 'params'), ctx=ctx)
    
    stages = [base_net.features[:6], base_net.features[6:8], base_net.features[8:10]]
    
    net = EfficientDet(base_size, stages, ratios, scales, steps, classes, 
                       fpn_channel=fpn_c, fpn_repeat=num_fpn, 
                       box_cls_repeat=box_cls_repeat, **kwargs)
    if pretrained:
        net.load_parameters(os.path.join(root, model_name + 'params'), ctx=ctx)

    return net

def efficientdet_b1_coco(pretrained_base=False, pretrained=False, **kwargs):
            
    from gluoncv.data import COCODetection
    classes = COCODetection.CLASSES
    net     = get_efficientdet('efficientdet-b1', classes, **kwargs)
    return net