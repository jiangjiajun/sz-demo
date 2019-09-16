import argparse
import functools
import distutils.util
import os

def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument. 

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# ENV
add_arg('use_gpu',                  bool,   True,                   "Whether to use GPU.")
add_arg('gpu_id',                   str,    '0',                   "Which GPU is used.")
add_arg('model_save_dir',           str,    "./output",        "The directory path to save model.")
add_arg('data_dir',                 str,    "./data/ILSVRC2012/",   "The ImageNet dataset root directory.")
add_arg('use_pretrained',           bool,    True,                   "Whether to use pretrained model.")
add_arg('checkpoint',               str,    None,                   "Whether to resume checkpoint.")
add_arg('save_step',                int,    1,                      "The steps interval to save checkpoints")
add_arg('do_val',       bool,  True,        "Whether to use the validation dataset")
# SOLVER AND HYPERPARAMETERS
add_arg('model',                    str,    "ResNet50",   "The name of network.")
add_arg('num_epochs',               int,    120,                    "The number of total epochs.")
add_arg('image_h',      int ,   224,          "input image h")
add_arg('image_w',      int ,   224,          "input image w")
add_arg('batch_size',               int,    8,                      "Minibatch size on a device.")
add_arg('lr',                       float,  0.1,                    "The learning rate.")
add_arg('lr_strategy',              str,    "piecewise_decay",      "The learning rate decay strategy.")
# READER AND PREPROCESS
add_arg('resize_short_size',        int,    256,                    "The value of resize_short_size")
add_arg('use_mixup',                bool,   False,                  "Whether to use mixup")
parser.add_argument('--image_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406], help="The mean of input image data")
parser.add_argument('--image_std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help="The std of input image data")
add_arg('use_label_smoothing',      bool,   False,                  "Whether to use label_smoothing")


args = parser.parse_args()
if args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
img_size = '3,' + str(args.image_h) + ',' + str(args.image_w)
data_dir= args.data_dir
train_txt = data_dir + '/train_list.txt'
if not os.path.exists(data_dir):
    print('[CHECK] ' + 'The dataset path is not exists!')
    exit(1)
if not os.path.exists(train_txt):
    print('[CHECK] ' + 'The train list file path is not exists!')
    exit(1)
label_list = []
img_len = 0
with open(train_txt) as flist:
    full_lines = [line.strip() for line in flist]
    img_len = len(full_lines)
    for line in full_lines:
        part = line.split(' ')
        label = int(part[-1])
        if label not in label_list:
            label_list.append(label)
label_dims = len(label_list)

if args.model == 'AlexNet' and args.use_pretrained:
    if not (args.image_h == 224 and args.image_w == 224):
        print('[CHECK] ' + 'The AlexNet\'s h and w must be 224!')
        exit(0)
elif args.model.startwiths('ResNet')  and args.use_pretrained:
    if not (args.image_h % 32 == 0 and args.image_w % 32 == 0):
        print('[CHECK] ' + 'This number of h and w must be divisible by 32')
settings = args
settings.class_dim = label_dims
settings.total_images = img_len
settings.print_step = 10
settings.test_batch_size = 8
settings.image_shape = img_size
settings.random_seed = None
settings.l2_decay = 1e-4
settings.momentum_rate = 0.9
settings.lower_scale = 0.08
settings.lower_ratio = 3./4.
settings.upper_ratio = 4./3.
settings.mixup_alpha = 0.2
settings.reader_thread = 8
settings.reader_buf_size = 2048
settings.interpolation = None
settings.label_smoothing_epsilon = 0.1
settings.step_epochs = [30, 60, 90]

# get pretrained model
pretrained_url = {
                'AlexNet': 'http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar',
                'MobileNetV1_x0_25': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_25_pretrained.tar',
                'MobileNetV1_x0_5': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_5_pretrained.tar',
                'MobileNetV1_x0_75': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_75_pretrained.tar',
                'MobileNetV1': 'http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar',
                'MobileNetV2_x0_25': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar',
                'MobileNetV2_x0_5': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar',
                'MobileNetV2_x0_75': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_75_pretrained.tar',
                'MobileNetV2': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar',
                'MobileNetV2_x1_5': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar',
                'MobileNetV2_x2_0': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar',
                'MobileNetV3_small_x1_0': 'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar',
                'ResNet101_vd': 'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar'
                 }
fc_name = {
            'AlexNet': ['fc8_offset', 'fc8_weights'],
            'MobileNetV1_x0_25': ['fc7_weights', 'fc7_offset'],
            'MobileNetV1_x0_5': ['fc7_weights', 'fc7_offset'],
            'MobileNetV1_x0_75': ['fc7_weights', 'fc7_offset'],
            'MobileNetV1': ['fc7_weights', 'fc7_offset'],
            'MobileNetV2_x0_25': ['fc10_weights', 'fc10_offset'],
            'MobileNetV2_x0_5': ['fc10_weights', 'fc10_offset'],
            'MobileNetV2_x0_75': ['fc10_weights', 'fc10_offset'],
            'MobileNetV2': ['fc10_weights', 'fc10_offset'],
            'MobileNetV2_x1_5': ['fc10_weights', 'fc10_offset'],
            'MobileNetV2_x2_0': ['fc10_weights', 'fc10_offset'],
            'MobileNetV3_small_x1_0': ['fc_weights', 'fc_offset'],
            'ResNet101_vd': ['fc_0.w_0', 'fc_0.b_0']
          }
if settings.use_pretrained and settings.checkpoint is None:
    part = pretrained_url[settings.model].split('/')
    path = './pretrained/' + part[-1][:-4]
    if not os.path.exists(path):
        import requests
        url_req = requests.get(pretrained_url[settings.model])
        with open('./pretrained/' + part[-1], 'wb') as fw:
            fw.write(url_req.content)
        import tarfile
        tar = tarfile.open('./pretrained/' + part[-1])
        names = tar.getnames()
        for name in names:
            tar.extract(name, path='./pretrained/')
        if settings.class_dim != 1000:
            os.remove('./pretrained/' + part[-1][:-4] + '/' + fc_name[settings.model][0])
            os.remove('./pretrained/' + part[-1][:-4] + '/' + fc_name[settings.model][1])
settings.pretrained_model = './pretrained/' + part[-1][:-4]

# start train
from train import *
try:
    main(settings)
except AssertionError as e:
    print('[CHECK] ' + str(e))
    exit(1)
# TODO：except--out of memory 

print('Train Over!')
exit(0)
    
