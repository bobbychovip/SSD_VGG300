# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2


from tensorflow.contrib import slim

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from preprocessing import img_preprocess
from ssd_vgg300 import SSDNet
from eval import np_methods
import visualization

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = img_preprocess.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=img_preprocess.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

reuse = True if 'ssd_net' in locals() else None
ssd = SSDNet()
with slim.arg_scope(ssd.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd.net(image_4d, is_training=False, reuse=reuse)

# 载入模型
isess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./checkpoints/ssd_vgg_300_weights.ckpt.meta')
saver.restore(isess, tf.train.latest_checkpoint('./checkpoints'))

ssd_anchors = ssd.anchors(net_shape)

def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    print(rpredictions)
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    print(rbboxes)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

path = './demo/'
image_name = '000001.jpg'

image_name = path + image_name

img = mpimg.imread(image_name)
rclasses, rscores, rbboxes = process_image(img)

#visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

#plt.show()
