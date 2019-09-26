中文 | [English](README_en.md)

# 图像分类以及模型库

## 内容
- [简介](#简介)
- [快速开始](#快速开始)
    - [安装说明](#安装说明)
    - [数据准备](#数据准备)
    - [模型训练](#模型训练)
    - [Auto FineTune的使用](#AutoFineTune的使用)
---

## 简介
图像分类是计算机视觉的重要领域，它的目标是将图像分类到预定义的标签。近期，许多研究者提出很多不同种类的神经网络，并且极大的提升了分类算法的性能。本页将介绍如何使用PaddlePaddle进行图像分类。

## 快速开始

### 安装说明

在当前目录下运行样例代码需要python 2.7及以上版本，PadddlePaddle Fluid v1.5.1或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据 [安装文档](http://paddlepaddle.org/documentation/docs/zh/1.5/beginners_guide/install/index_cn.html) 中的说明来更新PaddlePaddle。

#### 环境依赖

python >= 2.7，CUDA >= 8.0，CUDNN >= 7.0，Paddle >= 1.5.0
运行训练代码需要安装numpy，cv2
使用预训练模型需要requests

```bash
pip install paddlepaddle-gpu
pip install opencv-python
pip install numpy
pip install requests
```

### 数据准备
数据按照如下方式进行存放，其中train_image文件夹（可自定义命名）存放所有的图像数据，train_list.txt、val_list.txt、test_list.txt分别用于存放训练、验证、测试数据信息。
```
./mydataset/
  ├── train_image
  │   ├── IMG_3655.jpg
  │   ├── IMG_3656.jpg
  │   ├── IMG_3657.jpg
  |   ...
  ├── train_list.txt
  ├── val_list.txt
  ├── test_list.txt
```
其中，txt文件中存放相对于`./mydataset/`的图像相对路径和图像类别，并用空格分割开，详细如下：
```
./train_image/IMG_3655.jpg 0
./train_image/IMG_3656.jpg 2
./train_image/IMG_3657.jpg 1
...
```
### 模型训练

数据准备完毕后，可以通过如下的方式启动训练：
```
python run.py \
       --configs_yaml=./confifgs/demo.yaml  \
```
**参数说明：**

|参数名 | 类型 | 含义 | 默认值 | 
|---|---|---|---|
|configs_yaml<input type="checkbox" class="rowselector hidden"> | str | 存放参数的yaml文件路径 | ‘./confifgs/demo.yaml’ |

**yaml文件参数说明：**

|参数名 | 类型 | 含义 | 默认值 | 
|---|---|---|---|
|use_gpu<input type="checkbox" class="rowselector hidden"> | bool | 是否使用gpu | True | 
|gpu_id<input type="checkbox" class="rowselector hidden"> | str | 使用哪一张显卡（有且仅使用1张） | ’0‘ | 
|saved_params_dir<input type="checkbox" class="rowselector hidden"> | str | 模型保存路径 | ’./output‘ | 
|data_dir<input type="checkbox" class="rowselector hidden"> | str | 数据存放路径 | ’./data/ILSVRC2012/‘ | 
|use_pretrained<input type="checkbox" class="rowselector hidden"> | bool | 是否使用官方预训练模型 | True | 
|checkpoint<input type="checkbox" class="rowselector hidden"> | str | 自己的预训练模型（若此参数不为None，上一参数必须为False） | None | 
|save_step<input type="checkbox" class="rowselector hidden"> | int | 每隔多少轮保存模型 | 1 | 
|model<input type="checkbox" class="rowselector hidden"> | str | 使用哪一类模型 | ’ResNet50‘ | 
|num_epochs<input type="checkbox" class="rowselector hidden"> | int | 训练轮数 | 120 | 
|image_h<input type="checkbox" class="rowselector hidden"> | int | 图像高度 | 224 | 
|image_w<input type="checkbox" class="rowselector hidden"> | int | 图像宽度 | 224 | 
|batch_size<input type="checkbox" class="rowselector hidden"> | int | 批大小 | 256 | 
|lr<input type="checkbox" class="rowselector hidden"> | float | 学习率 | 0.1 | 
|lr_strategy<input type="checkbox" class="rowselector hidden"> | str | 学习策略 | ’piecewise_decay‘ | 
|resize_short_size<input type="checkbox" class="rowselector hidden"> | int | 短边resize的长度 | 256 | 
|use_default_mean_std<input type="checkbox" class="rowselector hidden"> | bool | 是否使用默认的均值和方差 | False | 
|use_distrot<input type="checkbox" class="rowselector hidden"> | bool | 是否使用数据扰动 | True | 
|use_rotate<input type="checkbox" class="rowselector hidden"> | bool | 是否使用图像旋转 | True | 

**数据读取器说明：** 数据读取器定义在```reader.py```文件中，现在默认基于cv2的数据读取器。当前支持的数据增广方式有：

* 旋转
* 颜色抖动（暂未实现）
* 随机裁剪
* 中心裁剪
* 长宽调整
* 水平翻转

【备注】

 1. 没有pretrained model的模型在选择完模型后默认为use_pretrained为False。
 2. 部分模型的size是固定的，在选择完模型并确定使用pretrained model后image_h和image_w固定位某个值。（如AlexNet）


### AutoFineTune的使用
***安装***：参考[PaddleHub安装教程](https://github.com/PaddlePaddle/PaddleHub/tree/develop)         
数据准备完毕后，可以通过如下的方式启动训练：
```
hub autofinetune run.py \
       --param_file=auto_finetune.yaml \
       --cuda=['2','3'] \
       --popsize=5 \
       --round=5 \
       --evaluate_choice=fulltrail \
       --tuning_strategy=HAZero \
       --output_dir=./output \
       configs_yaml ./confifgs/demo.yaml 
```
**参数说明：**

|参数名 | 类型 | 含义 | 默认值 | 
|---|---|---|---|
|param_file<input type="checkbox" class="rowselector hidden"> | str | yaml文件路径（AutoFineTune参数） | 此为固定值不可替换 | 
|cuda<input type="checkbox" class="rowselector hidden"> | list | 使用的gpu的卡的id（AutoFineTune参数） | ['0'] | 
|popsize<input type="checkbox" class="rowselector hidden"> | int | 每个round的组合数（AutoFineTune参数） | 此为固定值不可替换 | 
|round<input type="checkbox" class="rowselector hidden"> | int | auto finetune的轮数（AutoFineTune参数） | 此为固定值不可替换 | 
|evaluate_choice<input type="checkbox" class="rowselector hidden"> | str | 超参优化评价策略（AutoFineTune参数） | 此为固定值不可替换 | 
|tuning_strategy<input type="checkbox" class="rowselector hidden"> | str | 超参优化搜索策略（AutoFineTune参数） | 此为固定值不可替换 | 
|output_dir<input type="checkbox" class="rowselector hidden"> | str | 模型保存路径（AutoFineTune参数） | ’./output‘ |
|configs_yaml<input type="checkbox" class="rowselector hidden"> | str | 存放参数的yaml文件路径 | ‘./confifgs/demo.yaml’ |

**yaml文件参数说明：**

|参数名 | 类型 | 含义 | 默认值 | 
|---|---|---|---|
|data_dir<input type="checkbox" class="rowselector hidden"> | str | 数据存放路径 | ’./data/ILSVRC2012/‘ | 
|use_pretrained<input type="checkbox" class="rowselector hidden"> | bool | 是否使用官方预训练模型 | True | 
|use_auto_finetune<input type="checkbox" class="rowselector hidden"> | bool | 是否使用AutoFineTune | False | 
|checkpoint<input type="checkbox" class="rowselector hidden"> | str | 自己的预训练模型（若此参数不为None，上一参数必须为False） | None | 
|save_step<input type="checkbox" class="rowselector hidden"> | int | 每隔多少轮保存模型 | 1 | 
|model<input type="checkbox" class="rowselector hidden"> | str | 使用哪一类模型 | ’ResNet50‘ | 
|image_h<input type="checkbox" class="rowselector hidden"> | int | 图像高度 | 224 | 
|image_w<input type="checkbox" class="rowselector hidden"> | int | 图像宽度 | 224 | 
|lr_strategy<input type="checkbox" class="rowselector hidden"> | str | 学习策略 | ’piecewise_decay‘ | 
|resize_short_size<input type="checkbox" class="rowselector hidden"> | int | 短边resize的长度 | 256 | 
|use_default_mean_std<input type="checkbox" class="rowselector hidden"> | bool | 是否使用默认的均值和方差 | False | 
|use_distrot<input type="checkbox" class="rowselector hidden"> | bool | 是否使用数据扰动 | True | 
|use_rotate<input type="checkbox" class="rowselector hidden"> | bool | 是否使用图像旋转 | True | 
