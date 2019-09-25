Confusion Matrix
===

使用Confusion Matrix工具需要predict.txt和label.txt两个文件,其中predict.txt存储了预测类别和真实类别，label.txt存储了类别对应的名称，分别为如下格式

predict.txt：
---

0 1 \
1 1 \
2 4 \
4 1 \
...
 
其中每一行代表一次预测，第一列表示预测值，第二列表示真实值


label.txt
---
cat \
dog \
tiger \
...

其中每一行代表了label中的一个类别
 
