# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import sys
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# 原始数据
VOC_DIRECTORY = '/Volumes/Transcend/VOC2012/'
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# VOC_LABEL
VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

# TFRecords参数
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200 

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def process_image(directory, name):
    """
    :param directory: voc文件夹
    :param name: 图片名或xml文件名
    :return: 需要写入tfr的数据
    """
    # 读取图片
    # DIRECTORY_IMAGES = 'JPEGImages/'
    filename = os.path.join(directory + DIRECTORY_IMAGES + name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # 读取XML文件
    # DIRECTORY_ANNOTATIONS = 'Annotations/'
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # 图片形状
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    # Find annotations
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))
 
        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)
 
        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated

def convert_to_example(image_data, labels, labels_text, bboxes, shape,
                       difficult, truncated):
    """
    Args:
        image_data: string类型，RGB图像的JPEG编码
        labels: VOC_LABELS[label][0]构成的list，groundtruth标签
        labels_text: string 类型的list，groundtruth对应的人类可读的标签，就是VOC_LABELS[label][0]的一级索引label
        bboxes: groudtruth对应的真实框的坐标[xmin, ymin, xmax, ymax]
        shape: RGB图像的形状
    returns:
        Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [coord.append(point) for coord, point in zip([ymin, xmin, ymax, xmax], b)]
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),  # 图像编码格式
            'image/encoded': bytes_feature(image_data)}))  # 二进制图像数据
    return example

def add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """
    从JPEGImage文件夹和Annotations 文件夹读取数据并写入TFRcord
    
    Args:
        dataset_dir: VOC数据集的地址即VOC_DIRECTORY
        name: 图像jpeg 文件或xml文件的文件名，两者是相同的
        tfrecord_writer: 写TFRcord的对象
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        process_image(dataset_dir, name)
    example = convert_to_example(image_data, labels, labels_text,
                                 bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())

def get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def run(dataset_dir, output_dir, name='voc2012', shuffling=False):
    """ 生成TFRcord文件
    Args:
      dataset_dir: VOC数据集的地址即VOC_DIRECTORY
      output_dir: TFRcord文件的存放地址
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                if filename[0:2] == '._':
                    img_name = filename[2:-4]
                else:
                    img_name = filename[:-4]
                add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Pascal VOC dataset!')

#========================================
# DEBUG
#========================================
img_names = ['2007_000027', '2007_000032', '2007_000033']
for img_name in img_names:
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = process_image(VOC_DIRECTORY, img_name)
    with tf.Session() as sess:
        image_jpg = tf.image.decode_jpeg(image_data)
        print(sess.run(image_jpg))
        image_jpg = tf.image.convert_image_dtype(image_jpg, dtype=tf.uint8)

        plt.figure(1)
        plt.imshow(image_jpg.eval())
        plt.show()

        print(shape)
        print(bboxes)
        print(labels)
        print(labels_text)

run(VOC_DIRECTORY, './tfrecords', 'voc2012', False)

