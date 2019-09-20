from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import sys
import logging
import paddle
import models
import argparse
import functools
import paddle.fluid as fluid
import reader
from utility import add_arguments, print_arguments

from paddle.fluid.contrib.slim import Compressor

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64*4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('gpu_id',          str, '0',                "Which GPU is used.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_width',      str,  '224',         "Input image width")
add_arg('image_height',      str,  '224',         "Input image height")
add_arg('model',            str,  '',          "Set the network to use.")
add_arg('pretrained_model', str,  '',                "Whether to use pretrained model.")
add_arg('data_dir',       str, '',                "Data path of images.")
add_arg('target_ratio',       float, 0.5,                "Flops of prune.")
add_arg('strategy',       str, 'Uniform',                "Strategy of prune.")
add_arg('checkpoint_path', str, './checkpoints',       'Path of save model')
parser.add_argument('--img_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406], help="The mean of input image data")
parser.add_argument('--img_std', nargs='+', type=float, default=[0.229, 0.224, 0.225], help="The std of input image data")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]
strategy_list = ['Uniform', 'Sensitive']

def compress(args):
    assert args.batch_size > 0, 'batch size of input should be more than one'
    image_shape = [3, int(args.image_height) ,int(args.image_width)]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    assert args.strategy in strategy_list, "{} is not in lists: {}".format(args.strategy,
                                                                           strategy_list)
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()

    if args.model is "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=args.class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)
        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=args.class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    val_program = fluid.default_main_program().clone()

    assert os.path.exists(args.data_dir), "data directory '{}' is not exist".format(args.data_dir)
    train_file_list = os.path.join(args.data_dir, 'train_list.txt')
    assert os.path.exists(train_file_list), "data directory '{}' is not exist".format(train_file_list)       
    val_file_list = os.path.join(args.data_dir, 'val_list.txt')
    assert os.path.exists(val_file_list), "data directory '{}' is not exist".format(val_file_list)
    with open(train_file_list, 'r') as f:
        lines = f.readlines()
    total_images = len(lines)
    boundaries=[total_images / args.batch_size * 30,
                total_images / args.batch_size * 60,
                total_images / args.batch_size * 90]
    values=[0.001, 0.0001, 0.00001, 0.000001]
    opt = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=boundaries,
            values=values),
        regularization=fluid.regularizer.L2Decay(4e-5))

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    
    assert os.path.exists(args.pretrained_model), "pretrained model directory '{}' is not exist".format(args.pretrained_model) 
    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)
    val_reader = paddle.batch(reader.val(settings=args), batch_size=args.batch_size)
    val_feed_list = [('image', image.name), ('label', label.name)]
    val_fetch_list = [('acc_top1', acc_top1.name), ('acc_top5', acc_top5.name)]

    train_reader = paddle.batch(
        reader.train(settings=args), batch_size=args.batch_size, drop_last=True)
    train_feed_list = [('image', image.name), ('label', label.name)]
    train_fetch_list = [('loss', avg_cost.name)]

    teacher_programs = []
    distiller_optimizer = None

    com_pass = Compressor(
        place,
        fluid.global_scope(),
        fluid.default_main_program(),
        train_reader=train_reader,
        train_feed_list=train_feed_list,
        train_fetch_list=train_fetch_list,
        eval_program=val_program,
        eval_reader=val_reader,
        eval_feed_list=val_feed_list,
        eval_fetch_list=val_fetch_list,
        teacher_programs=teacher_programs,
        train_optimizer=opt,
        distiller_optimizer=distiller_optimizer)
    if args.strategy == 'Uniform':
        compress_config = "configs/filter_pruning_uniform.yaml"
    else:
        compress_config = "configs/filter_pruning_sen.yaml"
    com_pass.config(compress_config)
    assert args.target_ratio > 0 and args.target_ratio < 1, "prune ratio should be between 0 and 1"
    com_pass.strategies[0].target_ratio=args.target_ratio
    com_pass.checkpoint_path = args.checkpoint_path
    com_pass.run()
    pruned_prog = com_pass.eval_graph.program
    fluid.io.save_inference_model("./pruned_model/", [image.name], [out], exe, main_program=pruned_prog)


def main():
    args = parser.parse_args()
    print_arguments(args)
    try:
        compress(args)
    except AssertionError as e:
        print("[CHECK] ", e)
        exit(1) 


if __name__ == '__main__':
    main()
