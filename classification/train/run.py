# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import distutils.util
import os
import numpy as np
import cv2
import yaml
from types import SimpleNamespace


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + " Default: %(default)s.",
                           **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)

    # YAML PATH
    add_arg("configs_yaml", str, None,
            'The configure yaml file\'s path')
    # ENV
    add_arg("use_auto_finetune", bool, False, "Whether to use Auto Finetune.")
    add_arg("use_gpu", bool, True, "Whether to use GPU.")
    add_arg("gpu_id", str, "0", "Which GPU is used.")
    add_arg("saved_params_dir", str, "./output", "The directory path to save model.")
    add_arg(
        "data_dir", str, "./data/ILSVRC2012/", "The dataset root directory."
    )
    add_arg("use_pretrained", bool, True, "Whether to use pretrained model.")
    add_arg("checkpoint", str, None, "Whether to resume checkpoint.")
    add_arg("save_step", int, 1, "The steps interval to save checkpoints")
    # SOLVER AND HYPERPARAMETERS
    add_arg("model", str, "ResNet50", "The name of network.")
    add_arg("num_epochs", int, 120, "The number of total epochs.")
    add_arg("image_h", int, 224, "The input image h.")
    add_arg("image_w", int, 224, "The input image w.")
    add_arg("batch_size", int, 8, "Minibatch size on a device.")
    add_arg("lr", float, 0.1, "The learning rate.")
    add_arg("lr_strategy", str, "piecewise_decay", "The learning rate decay strategy.")
    # READER AND PREPROCESS
    add_arg("resize_short_size", int, 256, "The value of resize_short_size.")
    add_arg("use_default_mean_std", bool, False, "Whether to use label_smoothing.")
    
    args = parser.parse_args()
    yaml_path = args.configs_yaml
    if yaml_path is None or yaml_path == 'None':
        settings = args
    else:
        file = open(yaml_path)
        settings_dict = yaml.load(file, Loader=yaml.FullLoader)
        settings = SimpleNamespace(**settings_dict)
    if settings.checkpoint == "None":
        settings.checkpoint = None
    settings.gpu_id = str(settings.gpu_id)
    settings.model_save_dir = settings.saved_params_dir
    if not hasattr(settings, 'print_step'):
        settings.print_step = 10
    if not hasattr(settings, 'test_batch_size'):
        settings.test_batch_size = 8
    if not hasattr(settings, 'random_seed'):
        settings.random_seed = None
    if not hasattr(settings, 'l2_decay'):
        settings.l2_decay = 1e-4
    if not hasattr(settings, 'momentum_rate'):
        settings.momentum_rate = 0.9
    if not hasattr(settings, 'lower_scale'):
        settings.lower_scale = 0.08
    if not hasattr(settings, 'lower_ratio'):
        settings.lower_ratio = 3.0 / 4.0
    if not hasattr(settings, 'upper_ratio'):
        settings.upper_ratio = 4.0 / 3.0
    if not hasattr(settings, 'mixup_alpha'):
        settings.mixup_alpha = 0.2
    if not hasattr(settings, 'reader_thread'):
        settings.reader_thread = 8
    if not hasattr(settings, 'reader_buf_size'):
        settings.reader_buf_size = 2048
    if not hasattr(settings, 'interpolation'):
        settings.interpolation = 1
    if not hasattr(settings, 'label_smoothing_epsilon'):
        settings.label_smoothing_epsilon = 0.1
    if not hasattr(settings, 'step_epochs'):
        settings.step_epochs = [10, 20, 30, 40]
    else:
        str_step_epochs = settings.step_epochs
        part = str_step_epochs.split(',')
        settings.step_epochs = []
        for epoch in part:
            settings.step_epochs.append(int(epoch))
    if not hasattr(settings, 'use_mixup'):
        settings.use_mixup = False
    if not hasattr(settings, 'use_label_smoothing'):
        settings.use_label_smoothing = False
    if not hasattr(settings, 'use_distrot'):
        settings.use_distrot = True
    if not hasattr(settings, 'use_rotate'):
        settings.use_rotate = True

    # set the gpu
    if settings.use_gpu and not settings.use_auto_finetune:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu_id
    img_size = "3," + str(settings.image_h) + "," + str(settings.image_w)
    settings.image_shape = img_size

    # get the count of images, dims of label
    data_dir = settings.data_dir
    train_txt = os.path.join(data_dir, "train_list.txt")
    if not os.path.exists(data_dir):
        print("[CHECK] " + "The dataset path is not exists!")
        exit(1)
    if not os.path.exists(train_txt):
        print("[CHECK] " + "The train list file path is not exists!")
        exit(1)
    label_list = []
    img_len = 0
    with open(train_txt, "r") as flist:
        full_lines = [line.strip() for line in flist]
        img_len = len(full_lines)
        for line in full_lines:
            part = line.split(" ")
            label = int(part[-1])
            if label not in label_list:
                label_list.append(label)
    label_dims = len(label_list)
    settings.class_dim = label_dims
    settings.total_images = img_len

    # get the dataset's mean and std
    if not settings.use_default_mean_std:
        mean_std_path = os.path.join(settings.data_dir, "mean_std.txt")
        if os.path.exists(mean_std_path):
            with open(mean_std_path, "r") as flist:
                full_lines = [line.strip() for line in flist]
                line1 = full_lines[0].strip()[1:-1]
                list1 = line1.split(", ")
                settings.image_mean = []
                for s in list1:
                    settings.image_mean.append(float(s))
                line2 = full_lines[1].strip()[1:-1]
                list2 = line2.split(", ")
                settings.image_std = []
                for s in list2:
                    settings.image_std.append(float(s))
        else:
            print("[CHECK] " + "When don\'t use the default mean and std, you must calculate the mean and std firstly by the tools called cal_mean_std.")
            exit(1)
    else:
        settings.image_mean = [0.485, 0.456, 0.406]
        settings.image_std = [0.229, 0.224, 0.225]

    # check the image shape
    if not (settings.image_h <= settings.resize_short_size
            and settings.image_w <= settings.resize_short_size):
        print("[CHECK] " +
              "The image_h and image_w must be lower than resize_short_size!")
        exit(1)
    if settings.model == "AlexNet" and settings.use_pretrained:
        if not (settings.image_h == 224 and settings.image_w == 224):
            print("[CHECK] " + "The AlexNet's h and w must be 224!")
            exit(1)
    elif (settings.model.startswith("ResNet") and "_vd" in settings.model
          and settings.use_pretrained):
        if not (settings.image_h % 32 == 0 and settings.image_w % 32 == 0):
            print("[CHECK] " +
                  "This number of h and w must be divisible by 32")
            exit(1)

    # get pretrained model
    from pretrained_model_config import pretrained_url, fc_name

    if settings.use_pretrained and settings.checkpoint is None:
        part = pretrained_url[settings.model].split("/")
        path = os.path.join("./pretrained", part[-1][:-4])
        if not os.path.exists(path):
            print('Download the pretrained model. Wait for minutes...')
            import requests
            url_req = requests.get(pretrained_url[settings.model])
            with open(os.path.join("./pretrained", part[-1]), "wb") as fw:
                fw.write(url_req.content)
            import tarfile

            tar = tarfile.open(os.path.join("./pretrained", part[-1]))
            names = tar.getnames()
            for name in names:
                tar.extract(name, path="./pretrained")
            if settings.class_dim != 1000:
                os.remove(
                    os.path.join("./pretrained", part[-1][:-4],
                                 fc_name[settings.model][0]))
                os.remove(
                    os.path.join("./pretrained", part[-1][:-4],
                                 fc_name[settings.model][1]))
    settings.pretrained_model = os.path.join("./pretrained", part[-1][:-4])

    # start train
    from train import *

    try:
        out = main(settings)
        print("AutoFinetuneEval" + "\t" + str(out))
    except AssertionError as e:
        print("[CHECK] " + str(e))
        exit(1)
    # TODO：except--out of memory

    exit(0)
