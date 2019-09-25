1、Confusion Matrix
===
  在我们进行分类的时候，可以通过Confusion Matrix工具对分类结果进行分析，以改善我们的模型。

输入文件格式：
--
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
           python cm.py
 
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
          python kmean.py data_dir

其中data_dir代表了图像数据的路径
