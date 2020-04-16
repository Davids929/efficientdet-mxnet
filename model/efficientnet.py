#coding=utf-8
import mxnet as mx
import mxnet.gluon.nn as nn
from gluoncv.nn import ReLU6, HardSigmoid, HardSwish
import numpy as np
import math

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class Activation(nn.HybridBlock):
    """Activation function used in MobileNetV3"""
    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if act_func == "relu":
            self.act = nn.Activation('relu')
        elif act_func == "relu6":
            self.act = ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "swish":
            self.act = nn.Swish()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        elif act_func == "leaky":
            self.act = nn.LeakyReLU(alpha=0.375)
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, x):
        return self.act(x)

class _SE(nn.HybridBlock):
    def __init__(self, num_out, ratio=4, \
                 act_func=("relu", "hard_sigmoid"), use_bn=False, prefix='', **kwargs):
        super(_SE, self).__init__(**kwargs)
        self.use_bn = use_bn
        num_mid = make_divisible(num_out // ratio)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(channels=num_mid, \
                               kernel_size=1, use_bias=True, prefix=('%s_fc1_' % prefix))
        self.act1 = Activation(act_func[0])
        self.conv2 = nn.Conv2D(channels=num_out, \
                               kernel_size=1, use_bias=True, prefix=('%s_fc2_' % prefix))
        self.act2 = Activation(act_func[1])

    def hybrid_forward(self, F, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return F.broadcast_mul(x, out)

def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, act_type='swish', 
              norm_layer=nn.BatchNorm, norm_kwargs=None):

    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(norm_layer(scale=True, **norm_kwargs))
    if active:
        out.add(Activation(act_type))

class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, in_channels, channels, t, ksize, stride, use_se=True, 
                 act_type='swish', norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        pad = ksize//2
        with self.name_scope():
            self.out = nn.HybridSequential()
            if t != 1:
                _add_conv(self.out,
                          in_channels * t,
                          act_type=act_type,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out,
                      in_channels * t,
                      kernel=ksize,
                      stride=stride,
                      pad=pad,
                      num_group=in_channels * t,
                      act_type=act_type,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if use_se:
                self.out.add(_SE(in_channels * t))
            _add_conv(self.out,
                      channels,
                      active=False,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out

class EfficientNet(nn.HybridBlock):
    r"""EfficientNet model from the
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
    <https://arxiv.org/abs/1905.11946>`_ paper.

    """
    def __init__(self, w_multiplier=1.0, d_multiplier=1.0, dropout=1.0, classes=1000, 
                 use_se=True, act_type='swish', norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)
        with self.name_scope():
            in_channels_group = [make_divisible(x * w_multiplier) for x in [32, 16, 24, 40, 80, 112, 192]]
            channels_group = [make_divisible(x * w_multiplier) for x in [16, 24, 40, 80, 112, 192, 320]]
            num_repeat     = [math.ceil(x*d_multiplier) for x in [1, 2, 2, 3, 3, 4, 1]]
            ts = [1] + [6]*6
            ksizes  = [3, 3, 5, 3, 5, 5, 3]
            strides = [1, 2, 2, 2, 1, 2, 1]
            norm_kwargs = {} if norm_kwargs is None else norm_kwargs
            self.features = nn.HybridSequential()
            _add_conv(self.features, make_divisible(32 * w_multiplier), kernel=3,
                      stride=2, pad=1, act_type=act_type,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)

            for idx, (in_c, c, t, k, s, n) in enumerate(zip(in_channels_group, channels_group, 
                                                        ts, ksizes, strides, num_repeat)):    
                stage = nn.HybridSequential(prefix='stage%d_'%idx)
                for i in range(n):
                    stage.add(LinearBottleneck(in_channels=in_c,
                                               channels=c,
                                               t=t,
                                               ksize=k,
                                               stride=s,
                                               use_se=use_se,
                                               act_type=act_type,
                                               norm_layer=norm_layer,
                                               norm_kwargs=norm_kwargs))
                    in_c, s = c, 1
                self.features.add(stage)
            
            self.head   = nn.HybridSequential(prefix='head_')
            last_channels = make_divisible(1280 * w_multiplier) if w_multiplier > 1.0 else 1280
            with self.head.name_scope():
                _add_conv(self.head,
                          last_channels,
                          active=False, 
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(nn.GlobalAvgPool2D())
                self.output.add(nn.Dropout(dropout))
                self.output.add(nn.Dense(classes, use_bias=False, prefix='pred_'))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.head(x)
        x = self.output(x)
        return x


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]

def get_efficientnet(model_name, pretrained=False, ctx=mx.cpu(),
                     root='~/.mxnet/models', norm_layer=nn.BatchNorm, 
                     norm_kwargs=None, **kwargs):
    width, depth, res, dropout = efficientnet_params(model_name)
    net = EfficientNet(width, depth, dropout, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file(model_name, tag=pretrained, root=root), ctx=ctx)
    return net

def get_efficientnet_B0(**kwargs):
    return get_efficientnet('efficientnet-b0', **kwargs)

