#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import math
import random
import functools
import numpy as np
import cv2
import io
import signal

import paddle
import paddle.fluid as fluid
from PIL import Image, ImageEnhance

random.seed(0)
np.random.seed(0)


def rotate_image(img):
    """rotate image

    Args:
        img: image data

    Returns:
        rotated image data
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = np.random.randint(-10, 11)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def random_crop(img,
                size,
                settings,
                scale=None,
                ratio=None,
                interpolation=None):
    """random crop image

    Args:
        img: image data
        size: crop size
        settings: arguments
        scale: scale parameter
        ratio: ratio parameter

    Returns:
        random cropped image data
    """
    lower_scale = settings.lower_scale
    lower_ratio = settings.lower_ratio
    upper_ratio = settings.upper_ratio
    scale = [lower_scale, 1.0] if scale is None else scale
    ratio = [lower_ratio, upper_ratio] if ratio is None else ratio

    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[0]) / img.shape[1]) / (h**2),
                (float(img.shape[1]) / img.shape[0]) / (w**2))

    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * np.random.uniform(
        scale_min, scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)
    i = np.random.randint(0, img.shape[0] - h + 1)
    j = np.random.randint(0, img.shape[1] - w + 1)

    img = img[i:i + h, j:j + w, :]

    if interpolation:
        resized = cv2.resize(img, (size[2], size[1]),
                             interpolation=interpolation)
    else:
        resized = cv2.resize(img, (size[2], size[1]))
    return resized


#NOTE:(2019/08/08) distort color func is not implemented
def distort_color(img):
    """distort image color

    Args:
        img: image data

    Returns:
        distorted color image data
    """
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    def random_hue(img, lower=-18, upper=18):
        hue_delta = np.random.uniform(lower, upper)
        img = np.array(img.convert('HSV'))
        img[:, :, 0] = img[:, :, 0] + hue_delta
        img = Image.fromarray(img, mode='HSV').convert('RGB')
        return img

    if np.random.randint(0, 2) == 1:
        if np.random.randint(0, 2) == 1:
            ops = [
                random_brightness, random_contrast, random_color, random_hue
            ]
        else:
            ops = [
                random_brightness, random_color, random_hue, random_contrast
            ]
        #ops = [random_brightness, random_contrast, random_color]
        #np.random.shuffle(ops)
        img = Image.fromarray(img)
        img = ops[0](img)
        img = ops[1](img)
        img = ops[2](img)
        img = ops[3](img)
        img = np.asarray(img)
        return img
    else:
        return img


def resize_short(img, target_size, interpolation=None):
    """resize image

    Args:
        img: image data
        target_size: resize short target size
        interpolation: interpolation mode

    Returns:
        resized image data
    """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    if interpolation:
        resized = cv2.resize(img, (resized_width, resized_height),
                             interpolation=interpolation)
    else:
        resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    """crop image

    Args:
        img: images data
        target_size: crop target size
        center: crop mode

    Returns:
        img: cropped image data
    """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size[2]) // 2
        h_start = (height - size[1]) // 2
    else:
        w_start = np.random.randint(0, width - size[2] + 1)
        h_start = np.random.randint(0, height - size[1] + 1)
    w_end = w_start + size[2]
    h_end = h_start + size[1]
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def create_mixup_reader(settings, rd):
    """
    """
    class context:
        tmp_mix = []
        tmp_l1 = []
        tmp_l2 = []
        tmp_lam = []

    batch_size = settings.batch_size
    alpha = settings.mixup_alpha

    def fetch_data():

        data_list = []
        for i, item in enumerate(rd()):
            data_list.append(item)
            if i % batch_size == batch_size - 1:

                yield data_list
                data_list = []

    def mixup_data():
        for data_list in fetch_data():
            if alpha > 0.:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.
            l1 = np.array(data_list)
            l2 = np.random.permutation(l1)
            mixed_l = [
                l1[i][0] * lam + (1 - lam) * l2[i][0] for i in range(len(l1))
            ]
            yield (mixed_l, l1, l2, lam)

    def mixup_reader():
        for context.tmp_mix, context.tmp_l1, context.tmp_l2, context.tmp_lam in mixup_data(
        ):
            for i in range(len(context.tmp_mix)):
                mixed_l = context.tmp_mix[i]
                l1 = context.tmp_l1[i]
                l2 = context.tmp_l2[i]
                lam = context.tmp_lam
                yield (mixed_l, int(l1[1]), int(l2[1]), float(lam))

    return mixup_reader


def process_image(sample, settings, mode, color_jitter, rotate):
    """ process_image """

    mean = settings.image_mean
    std = settings.image_std
    crop_size = (3, settings.image_h, settings.image_w)
    img_path = sample[0]
    img = cv2.imread(img_path)

    if mode == 'train':
        if rotate:
            img = rotate_image(img)
        if crop_size[1] > 0 and crop_size[2] > 0:
            img = random_crop(img, crop_size, settings)
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            img = img[:, ::-1, :]
    else:
        if crop_size[1] > 0 and crop_size[2] > 0:
            target_size = settings.resize_short_size
            img = resize_short(img, target_size)
            img = crop_image(img, target_size=crop_size, center=True)

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return (img, sample[1])
    elif mode == 'test':
        return (img, )


def _reader_creator(settings,
                    file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=None):
    def reader():
        with open(file_list) as flist:
            full_lines = [line.strip() for line in flist]
            if mode != "test" and len(full_lines) < settings.batch_size:
                print(
                    "Warning: The number of the whole data ({}) is smaller than the batch_size ({}), and drop_last is turnning on, so nothing  will feed in program, Terminated now. Please reset batch_size to a smaller number or feed more data!"
                    .format(len(full_lines), settings.batch_size))
                os._exit(1)

            if shuffle:
                np.random.shuffle(full_lines)
        for line in full_lines:
            img_path, label = line.split()
            img_path = os.path.join(data_dir, img_path)
            if not os.path.exists(img_path):
                print("Warning: {} doesn't exist!".format(img_path))
            if mode == "train" or mode == "val":
                yield img_path, int(label)
            elif mode == "test":
                yield [img_path]

    mapper = functools.partial(process_image,
                               settings=settings,
                               mode=mode,
                               color_jitter=color_jitter,
                               rotate=rotate)

    return paddle.reader.xmap_readers(mapper,
                                      reader,
                                      settings.reader_thread,
                                      settings.reader_buf_size,
                                      order=False)


def train(settings):
    """Create a reader for trainning

    Args:
        settings: arguments

    Returns:
        train reader
    """
    file_list = os.path.join(settings.data_dir, 'train_list.txt')
    assert os.path.isfile(
        file_list
    ), "{} doesn't exist, please check train data list path".format(file_list)
    reader = _reader_creator(settings,
                             file_list,
                             'train',
                             shuffle=True,
                             color_jitter=settings.use_distrot,
                             rotate=settings.use_rotate,
                             data_dir=settings.data_dir)

    if settings.use_mixup == True:
        reader = create_mixup_reader(settings, reader)
    return reader


def val(settings):
    """Create a reader for eval

    Args:
        settings: arguments

    Returns:
        eval reader
    """
    file_list = os.path.join(settings.data_dir, 'val_list.txt')
    assert os.path.isfile(
        file_list), "{} doesn't exist, please check val data list path".format(
            file_list)

    return _reader_creator(settings,
                           file_list,
                           'val',
                           shuffle=False,
                           data_dir=settings.data_dir)


def test(settings):
    """Create a reader for testing

    Args:
        settings: arguments

    Returns:
        test reader
    """
    file_list = os.path.join(settings.data_dir, 'val_list.txt')
    assert os.path.isfile(
        file_list
    ), "{} doesn't exist, please check test data list path".format(file_list)
    return _reader_creator(settings,
                           file_list,
                           'test',
                           shuffle=False,
                           data_dir=settings.data_dir)
