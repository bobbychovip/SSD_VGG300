# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import tensorflow as tf

slim = tf.contrib.slim

FILE_PATTERN = 'voc2012_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

NUM_CLASSES = 21
NUM_SAMPLES = 17125 

def get_split(tfr_path, tfr_pattern=FILE_PATTERN, num_classes=NUM_CLASSES, num_samples=NUM_SAMPLES): 
    
# ===============TFR文件名匹配模板===============
    tfr_pattern = os.path.join(tfr_path, tfr_pattern)

    # =========阅读器=========
    reader = tf.TFRecordReader

    # ===================解码器===================
    keys_to_features = {  # 解码TFR文件方式
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {  # 解码二进制数据
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # =======描述字段=======
    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'shape': 'Shape of the image',
        'object/bbox': 'A list of bounding boxes, one per each object.',
        'object/label': 'A list of labels, one per each object.',
    }

    return slim.dataset.Dataset(
        data_sources=tfr_pattern,                     # TFR文件名
        reader=reader,                                # 阅读器
        decoder=decoder,                              # 解码器
        num_samples=num_samples,                      # 数目
        items_to_descriptions=items_to_descriptions,  # decoder条目描述字段
        num_classes=num_classes,                      # 类别数
        labels_to_names=None                          # 字典{图片:类别,……}
    )

def tfr_read(dataset):
    # 涉及队列操作，本部使用CPU设备
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,         # DatasetDataProvider 需要 slim.dataset.Dataset 做参数
        num_readers=2,
        common_queue_capacity=20 * 5,
        common_queue_min=10 * 5,
        shuffle=True)
    image, glabels, gbboxes = provider.get(['image',
                                            'object/label',
                                            'object/bbox'])
    return image, glabels, gbboxes

#=====================================
# DEBUG
#=====================================
"""
TFR_DIR = './tfrecords'
dataset = get_split(TFR_DIR, FILE_PATTERN, num_classes=21, num_samples=NUM_SAMPLES)
image, glabels, gbboxes = tfr_read(dataset)

with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run([glabels, gbboxes]))
    coord.request_stop()
    coord.join(threads)
"""
