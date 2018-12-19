# -*- coding: utf-8 -*-
#!/usr/bin/env python

from dataset import pascalvoc_2012
from preprocessing import img_preprocess

DATASET_DIR = './dataset/tfrecords'
BATCH_SIZE = 16
num_samples_per_epoch = 17125

dataset = pascalvoc_2012.get_split('./dataset/tfrecords',
                                   'voc2012_*.tfrecord',
                                   num_classes=21,
                                   num_samples=num_samples_per_epoch)
image, glabels, gbboxes = pascalvoc_2012.tfr_read(dataset)
print('Dataset:', dataset.data_sources, '|', dataset.num_samples)
image , glabels, gbboxes = img_preprocess.preprocess_image(image,
                                                          glabels,
                                                          gbboxes,
                                                          out_shape=(300, 300),
                                                          data_format='NCHW',
                                                          is_training=True)
print image.shape, glabels.shape, gbboxes.shape
