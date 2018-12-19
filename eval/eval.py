# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf

import ssd_util
from preprocessing import img_preprocess
from dataset import pascalvoc_2012
from ssd_vgg300 import SSDNet

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        ssd = SSDNet()
        ssd_shape = ssd.params.img_shape
        ssd_anchors = ssd.anchors(ssd_shape)

        

