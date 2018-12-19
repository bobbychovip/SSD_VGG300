# SSD_VGG300
A tensorflow implementation of Single Shot Multibox Detector.
# 使用
## VOC2012数据
下载VOC2012数据，并解压到文件夹VOC2012。更改pascalvoc_to_tfrecords.py文件的VOC_DIRECTORY地址为刚刚解压的文件夹的绝对地址。
## 生成TFR压缩数据
运行pascalvoc_to_tfrecords.py脚本生成TFR格式压缩的图片。
## checkpoint文件
下载checkpoint压缩文件并解压至checkpoint目录
## 训练模型
运行脚本train.py
## 测试模型
把检测图片放入demo文件夹，更改脚本ssd_demo.py中的图片路径
