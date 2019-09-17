# 模型压缩

## 内容
- [快速开始](#快速开始)
    - [安装说明](#安装说明)
    - [数据准备](#数据准备)
    - [模型训练](#模型训练)
---

## 快速开始

### 安装说明

在当前目录下运行样例代码需要python 2.7及以上版本，PadddlePaddle Fluid v1.5.1或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据 [安装文档](http://paddlepaddle.org/documentation/docs/zh/1.5/beginners_guide/install/index_cn.html) 中的说明来更新PaddlePaddle。

#### 环境依赖

python >= 2.7，CUDA >= 8.0，CUDNN >= 7.0
运行训练代码需要安装numpy，cv2

```bash
pip install opencv-python
pip install numpy
```

### 数据准备

首先，通过如下的方式进行数据的准备：


**步骤一：** 首先将所有图片保存在文件夹中， 如data/image。

**步骤二：** 在data中新建train_list.txt和val_list.txt

* train_list.txt: 训练集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
data/image/1.jpg 369
```
* val_list.txt: 验证集合的标签文件，每一行采用"空格"分隔图像路径与标注，例如：
```
data/image/2.jpg 65
```

### 模型训练

数据准备完毕后，可以通过如下的方式启动模型压缩：
```
python compress.py 
       --data_dir zhijian/ 
       --pretrained_model MobileNetV1_pretrained 
       --model MobileNet 
       --class_dim 1000 
       --image_width 224 
       --image_height 224 
       --batch_size 256 
       --target_ratio 0.5 
       --strategy Uniform 
       --use_gpu True 
       --gpu_id 0
       --img_mean 0.485 0.456 0.406
       --img_std 0.229 0.224 0.225

```


**参数说明：**

环境配置部分：

* **data_dir**: 数据存储路径，默认值: None
* **pretrained_model**: 加载预训练模型路径，默认值: None

模型类型和超参配置：

* **model**: 模型名称， 默认值: "MobileNet"
* **class_dim**: 类别数，默认值: 1000
* **image_shape**: 图片大小，默认值: "3,224,224"
* **batch_size**: batch size大小(所有设备)，默认值: 64
* **target_ratio**: 裁剪flop数比例， 默认值: 0.5
* **strategy**: 使用的裁剪策略， 包括Uniform, Sensitive，默认值：Uniform

预处理配置：

* **image_mean**: 图片均值，默认值：[0.485, 0.456, 0.406]
* **image_std**: 图片标准差，默认值：[0.229, 0.224, 0.225]

