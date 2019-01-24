# SSD_VGG300
A tensorflow implementation of Single Shot Multibox Detector.
## 使用
### VOC2012数据
下载KITTI数据集，将其转化为VOC2012格式，记录该文件夹的绝对地址。更改pascalvoc_to_tfrecords.py文件中VOC_DIRECTORY的值为刚刚解压后的文件夹的地址。
### 生成TFR压缩数据
运行pascalvoc_to_tfrecords.py脚本，把图片转换成TFR格式，方便tensorflow读取。
### 下载checkpoint文件
下载checkpoint压缩文件并解压至checkpoint目录。解压后的文件应包括类似ssd_vgg_300_weights.ckpt.data-00000-of-00001、ssd_vgg_300_weights.ckpt.index、ssd_vgg_300_weights.ckpt.meta、checkpoint的文件。
### 训练模型
运行脚本train.py，可根据训练状况调整max_steps、batch_size、adam_beta1、adam_beta2、opt_epsilon、num_epochs_per_decay等参数的值。
### 测试模型
把检测图片放入demo文件夹，更改脚本ssd_demo.py中的图片路径，运行该脚本即可得到结果。
