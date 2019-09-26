'''compute confusion matrix
labels.txt: contain label name.
predict.txt: predict_label true_label
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
import functools
import cv2
import os
import distutils.util
from sklearn.metrics import confusion_matrix


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

add_arg('output_dir',       str,  'None',                 "Directory of output")
add_arg('img_h',       int,  '2000',                 "Height of output image")
add_arg('img_w',       int,  '2000',                 "Width of output image")
add_arg('label_file',       str,  'None',                 "Directory of label.txt")
add_arg('predict_file',       str,  'None',                 "Directory of predict.txt")


def start_cm(args):
    # load labels.
    labels = []
    file = open(args.label_file, 'r')
    lines = file.readlines()
    for line in lines:
        labels.append(line.strip())
    file.close()

    y_true = []
    y_pred = []

    # load true and predict labels.
    file = open(args.predict_file, 'r')
    lines = file.readlines()
    for line in lines:
        y_true.append(int(line.split(" ")[1].strip()))
        y_pred.append(int(line.split(" ")[0].strip()))
    file.close()

    tick_marks = np.array(range(len(labels))) + 0.5

    def plot_confusion_matrix(cm, cmap=plt.cm.summer):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Predict label', y=1.04)
        # fig,ax=plt.subplots(1,1)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Confusion Matrix', fontsize=18)

    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_n = cm.astype('float')
    plt.figure(figsize=(10, 10), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_n[y_val][x_val]
        plt.text(x_val, y_val, "%0.2d" % (c,), color='red', fontsize=14, va='center', ha='center')

    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().invert_yaxis()
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_n)
    img_path = os.path.join(args.output_dir, 'cm.jpg')
    plt.savefig(img_path)
    cm_img = cv2.imread(img_path)
    cm_img = cv2.resize(cm_img, (args.img_w, args.img_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(img_path, cm_img)

def main():
    args = parser.parse_args()
    start_cm(args)


if __name__ == '__main__':
    main()

