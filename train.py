# -*- coding: utf-8 -*-
#!/usr/bin/env python

import tensorflow as tf

import ssd_util
from preprocessing import img_preprocess
from dataset import pascalvoc_2012
from ssd_vgg300 import SSDNet

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'

# =============================================== #
def main():
    max_steps = 30
    batch_size = 32
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    opt_epsilon = 1.0
    num_epochs_per_decay = 2.0
    num_samples_per_epoch = 17125
    moving_average_decay = None

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        with tf.device("device:CPU:0"):
            global_step = tf.train.create_global_step()
        
        ssd = SSDNet()
        ssd_shape = ssd.params.img_shape
        ssd_anchors = ssd.anchors(ssd_shape)

        dataset = pascalvoc_2012.get_split('./dataset/tfrecords',
                                           'voc2012_*.tfrecord',
                                           num_classes=21,
                                           num_samples=num_samples_per_epoch)

        with tf.device("/device:CPU:0"):
            image, glabels, gbboxes = pascalvoc_2012.tfr_read(dataset)
            image, glabels, gbboxes = img_preprocess.preprocess_image(image,
                                                                      glabels,
                                                                      gbboxes,
                                                                      out_shape=ssd_shape,
                                                                      data_format=DATA_FORMAT, is_training=True)
            gclasses, glocalisations, gscores = ssd.bboxes_encode(glabels, gbboxes, ssd_anchors)
            batch_shape = [1] + [len(ssd_anchors)] * 3

            r = tf.train.batch(ssd_util.reshape_list([image, gclasses, glocalisations, gscores]),
                               batch_size=batch_size,
                               num_threads=4,
                               capacity=5*batch_size)
            batch_queue = slim.prefetch_queue.prefetch_queue(r, capacity=2*1)
            b_image, b_gclasses, b_glocalisations, b_gscores = ssd_util.reshape_list(batch_queue.dequeue(), batch_shape)
            
            arg_scope = ssd.arg_scope(weight_decay=0.00004, data_format=DATA_FORMAT)
            with slim.arg_scope(arg_scope):
                predictions, localisations, logits, end_points = \
                    ssd.net(b_image, is_training=True)
            ssd.losses(logits, localisations,
                       b_gclasses, b_glocalisations, b_gscores,
                       match_threshold=.5,
                       negative_ratio=3,
                       alpha=1,
                       label_smoothing=0.)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if moving_average_decay:
                moving_average_variables = slim.get_model_variables()
                variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,
                                                                      global_step)
            else:
                moving_average_variables, variable_averages = None, None

            with tf.device("/device:CPU:0"):
                decay_steps = int(num_samples_per_epoch/batch_size*num_epochs_per_decay)
                learning_rate = tf.train.exponential_decay(0.01,
                                                           global_step,
                                                           decay_steps,
                                                           0.94,
                                                           staircase=True,
                                                           name='exponential_decay_learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate,
                                                   beta1=adam_beta1,
                                                   beta2=adam_beta2,
                                                   epsilon=opt_epsilon)
                tf.summary.scalar('learning_rate', learning_rate)

            if moving_average_decay:
                update_ops.append(variable_averages.apply(moving_average_variables))

            trainable_scopes = None
            if trainable_scopes is None:
                variables_to_train = tf.trainable_variables()
            else:
                scopes = [scope.strip() for scope in trainable_scopes.split(',')]
                variables_to_train = []
                for scope in scopes:
                    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    variables_to_train.extend(variables)

            losses = tf.get_collection(tf.GraphKeys.LOSSES)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses)
            loss = tf.add_n(losses)
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("regularization_loss", regularization_loss)

            grad = optimizer.compute_gradients(loss, var_list=variables_to_train)
            grad_updates = optimizer.apply_gradients(grad,
                                                     global_step=global_step)
            update_ops.append(grad_updates)

            with tf.control_dependencies(update_ops):
                total_loss = tf.add_n([loss, regularization_loss])
            tf.summary.scalar("total_loss", total_loss)

            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=5,
                                   keep_checkpoint_every_n_hours=1.0,
                                   write_version=2,
                                   pad_step_number=False)

            if True:
                import os
                import time

                print('start......')
                model_path = './checkpoints'
                model_file = tf.train.latest_checkpoint('./checkpoints/ssd_vgg_300_weights.ckpt.data-00000-of-00001')
                batch_size = batch_size
                with tf.Session() as sess:
                    summary = tf.summary.merge_all()
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    writer = tf.summary.FileWriter(model_path, sess.graph)

                    init_op = tf.group(tf.global_variables_initializer(), 
                                       tf.local_variables_initializer())
                    ckpt = tf.train.get_checkpoint_state(os.path.dirname('model_file'))
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)

                    init_op.run()
                    for step in range(max_steps):
                        start_time = time.time()
                        loss_value = sess.run(total_loss)

                        duration = time.time() - start_time
                        if step % 5 == 0:
                            summary_str = sess.run(summary)
                            writer.add_summary(summary_str, step)

                            example_per_sec = batch_size / duration
                            sec_per_batch = float(duration)
                            format_str = "[*] step %d, loss=%.2f(%.1f examples/sec; %.3f sec/batch)"
                            print(format_str%(step, loss_value, example_per_sec, sec_per_batch))
                        if step % 10 == 0 and step != 0:
                            saver.save(sess, os.path.join(model_path, "ssd_tf.model"), global_step=step)
                    coord.request_stop()
                    coord.join(threads)

if __name__ == '__main__':
    main()







