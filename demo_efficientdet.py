"""efficientdet Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from matplotlib import pyplot as plt
from model.efficientdet import get_efficientdet

def load_img(img_path, img_shape=512, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    img = mx.nd.imread(img_path)
    img = mx.nd.image.resize(img, (img_shape, img_shape))
    orig_img = img.asnumpy().astype('uint8')
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    img = img.expand_dims(0)
    return img, orig_img

def parse_args():
    parser = argparse.ArgumentParser(description='Test with efficientdet networks.')
    parser.add_argument('--network', type=str, default='efficientdet-b1',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=640,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Testing dataset. Now support voc.')
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained_path', type=str,
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold of object score when visualize the bboxes.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    if not args.images.strip():
        gcv.utils.download("https://cloud.githubusercontent.com/assets/3307514/" +
            "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg", 'street.jpg')
        image_list = ['street.jpg']
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.dataset.lower() == 'coco':
        from gluoncv.data import COCODetection
        classes = COCODetection.CLASSES
    elif args.dataset.lower() == 'voc':
        from gluoncv.data import VOCDetection
        classes = VOCDetection.CLASSES
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(args.dataset))

    net = get_efficientdet(args.network, classes, pretrained_base=False)
    net.load_parameters(args.pretrained_path)
    net.set_nms(0.45, 200)
    net.collect_params().reset_ctx(ctx = ctx)

    for image in image_list:
        ax = None
        x, img = load_img(image, short=args.data_shape)
        x = x.as_in_context(ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids, thresh=args.thresh,
                                    class_names=net.classes, ax=ax)
        plt.show()
