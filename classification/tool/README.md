1、Confusion Matrix
===
  在我们进行分类的时候，可以通过Confusion Matrix工具对分类结果进行分析，以改善我们的模型。

安装依赖：
----
        pip install sklearn
        pip install matplotlib
        pip install numpy
        
输入文件格式：
---
使用Confusion Matrix工具需要predict.txt和label.txt两个文件,其中predict.txt存储了预测类别和真实类别，label.txt存储了类别对应的名称，分别为如下格式

predict.txt：
----

        0 1 
        1 1 
        2 4 
        4 1 
        ...
 
其中每一行代表一次预测，第一列表示预测值，第二列表示真实值


label.txt
----
        cat 
        dog 
        tiger 
        ...

其中每一行代表了label中的一个类别
 
 使用方式：
 ----
           python cm.py --label_file ./label.txt --predict_file ./predict.txt --output_dir output --img_h 1000 --img_w 1000
  
 其中label_file指的是label.txt的路径，predict_file指的是predict.txt的路径，output_dir指的是输出混淆矩阵图片的保存路径，img_h指的是保存混淆矩阵图片的高，img_w指的是保存混淆矩阵图片的宽。
 
2、Kmeans 聚类
===

  很多情况下每一类别中的数据会出现很多重复的状况，这样会对模型造成影响，因此可以尝试使用kmeans对图片进行聚类，筛去重复的图片。
  
安装依赖：
----
        pip install opencv-python
        pip install scipy
        pip install numpy
        pip install pickle
        pip install matplotlib
        
除以上依赖外，还需要安装PCV：

        git clone https://github.com/jesolem/PCV
        cd PCV
        python setup.py install
使用方式：
----
          python kmean.py --data_dir data/image  --output_dir output --num_class 6

其中data_dir指的是图像数据的路径, output_dir指的是聚类后图片保存的路径，num_class 指的是想要聚类的类别数。
