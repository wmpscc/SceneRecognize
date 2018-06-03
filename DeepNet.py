#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import numpy as np
import os
import time

BATCH_SIZE = 10
slim = tf.contrib.slim


# 读训练集数据
def read_train_data():
    reader = tf.TFRecordReader()
    filename_train = tf.train.string_input_producer(["TFRecord/train.tfrecords"])
    _, serialized_example_test = reader.read(filename_train)
    features = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    img_train = features['image_raw']
    images_train = tf.decode_raw(img_train, tf.uint8)
    images_train = tf.reshape(images_train, [229, 229, 3])
    labels_train = tf.cast(features['label'], tf.int64)
    labels_train = tf.cast(labels_train, tf.int64)
    labels_train = tf.one_hot(labels_train, 80)
    return images_train, labels_train


# 读测试集数据
def read_test_data():
    reader = tf.TFRecordReader()
    filename_test = tf.train.string_input_producer(["TFRecord/test.tfrecords"])
    _, serialized_example_test = reader.read(filename_test)
    features_test = tf.parse_single_example(
        serialized_example_test,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    img_test = features_test['image_raw']
    images_test = tf.decode_raw(img_test, tf.uint8)
    images_test = tf.reshape(images_test, [229, 229, 3])
    labels_test = tf.cast(features_test['label'], tf.int64)
    labels_test = tf.one_hot(labels_test, 80)
    return images_test, labels_test


def save_model(sess, step):
    MODEL_SAVE_PATH = "model/"
    MODEL_NAME = "model.ckpt"
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)


def get_accuracy(logits, label):
    print(logits)
    logits = tf.reshape(logits, shape=[BATCH_SIZE, 80])
    current = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), 'float')
    accuracy = tf.reduce_mean(current)
    return accuracy


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')

        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


def net(inputs, dropout_keep_prob):
    # 229*229*3
    net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True}, padding='VALID')
    net = slim.max_pool2d(net, 2, stride=2)

    net = slim.conv2d(net, 64, 1, padding='SAME')
    net = slim.repeat(net, 2, slim.conv2d, 128, 3, normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True}, padding='VALID')
    net = slim.max_pool2d(net, 2, stride=2)

    net = slim.conv2d(net, 128, 1, padding='SAME')
    net = slim.repeat(net, 3, slim.conv2d, 256, 3, normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True}, padding='VALID')
    net = slim.max_pool2d(net, 2, stride=2)

    net = slim.conv2d(net, 256, 1, padding='SAME')
    net = slim.repeat(net, 3, slim.conv2d, 512, 3, normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True}, padding='VALID')
    net = slim.max_pool2d(net, 2, stride=2)

    net = slim.conv2d(net, 512, 1, padding='SAME')
    net = slim.repeat(net, 3, slim.conv2d, 512, 3, normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': True}, padding='VALID')
    # net = slim.max_pool2d(net, 2, stride=2)

    # net = slim.repeat(net, 2, slim.conv2d, 512, 3, normalizer_fn=slim.batch_norm,
    #                   normalizer_params={'is_training': True}, padding='SAME')
    # net = slim.conv2d(net, 80, 1, scope='f1')
    # net = tf.reduce_mean(net, [1, 2], keep_dims=True)
    net = slim.flatten(net)
    fc1 = slim.fully_connected(net, 80, activation_fn=None)

    return fc1


def net2(inputs, dropout_keep_prob):
    net = slim.conv2d(inputs, 32, 3, stride=2, padding='SAME')
    net = slim.conv2d(net, 32, 3, stride=1, padding='VALID')
    net = slim.conv2d(net, 64, 3, stride=1, padding='SAME')
    net = slim.batch_norm(net, is_training=True)
    with tf.variable_scope('FilterConcat1'):
        with tf.variable_scope('Branch0'):
            branch_0 = slim.max_pool2d(net, 3, stride=2, padding='VALID')
        with tf.variable_scope('Branch1'):
            branch_1 = slim.conv2d(net, 96, 3, stride=2, padding='VALID')
        net = tf.concat([branch_0, branch_1], 3)
    with tf.variable_scope('FilterConcat2'):
        with tf.variable_scope('Branch0'):
            branch0_0 = slim.conv2d(net, 64, 1, padding='SAME')
            branch0_1 = slim.conv2d(branch0_0, 96, 3, padding='VALID')
        with tf.variable_scope('Branch1'):
            branch1_0 = slim.conv2d(net, 64, 1, padding='SAME')
            branch1_1 = slim.conv2d(branch1_0, 64, [7, 1], padding='SAME')
            branch1_2 = slim.conv2d(branch1_1, 64, [1, 7], padding='SAME')
            branch1_3 = slim.conv2d(branch1_2, 96, 3, padding='VALID')
        net = tf.concat([branch0_1, branch1_3], 3)
    with tf.variable_scope('FilterConcat3'):
        with tf.variable_scope('Branch0'):
            branch_0 = slim.conv2d(net, 192, 3, stride=2, padding='VALID')
        with tf.variable_scope('Branch1'):
            branch_1 = slim.max_pool2d(net, 3, stride=2, padding='VALID')
        net = tf.concat([branch_0, branch_1], 3)
        net = slim.batch_norm(net, is_training=True)

    # 35*35*384
    net = slim.repeat(net, 5, block35, scale=0.34, activation_fn=tf.nn.leaky_relu)
    net = slim.batch_norm(net, is_training=True)

    with tf.variable_scope('FilterConcat4'):
        with tf.variable_scope('Branch0'):
            branch0 = slim.max_pool2d(net, 3, stride=2, padding='VALID')
        with tf.variable_scope('Branch1'):
            branch1 = slim.conv2d(net, 384, 3, stride=2, padding='VALID')
        with tf.variable_scope('Branch2'):
            branch2_0 = slim.conv2d(net, 256, 1, stride=2, padding='SAME')
            branch2_1 = slim.conv2d(branch2_0, 256, 3, padding='SAME')
            branch2_2 = slim.conv2d(branch2_1, 384, 2, padding='VALID')
        net = tf.concat([branch0, branch1, branch2_2], 3)
        net = slim.batch_norm(net, is_training=True)
    # 12*12
    net = slim.repeat(net, 10, block17, scale=0.20, activation_fn=tf.nn.leaky_relu)
    net = slim.batch_norm(net, is_training=True)

    with tf.variable_scope('FilterConcat5'):
        with tf.variable_scope('Branch0'):
            branch0 = slim.max_pool2d(net, 3, stride=2, padding='VALID')
        with tf.variable_scope('Branch1'):
            branch1_0 = slim.conv2d(net, 256, 1, padding='SAME')
            branch1_1 = slim.conv2d(branch1_0, 384, 3, stride=2, padding='VALID')
        with tf.variable_scope('Branch2'):
            branch2_0 = slim.conv2d(net, 256, 1, padding='SAME')
            branch2_1 = slim.conv2d(branch2_0, 288, 3, stride=2, padding='VALID')
        with tf.variable_scope('Branch3'):
            branch3_0 = slim.conv2d(net, 256, 1, padding='SAME')
            branch3_1 = slim.conv2d(branch3_0, 288, 3, padding='SAME')
            branch3_2 = slim.conv2d(branch3_1, 320, 3, stride=2, padding='VALID')
        net = tf.concat([branch0, branch1_1, branch2_1, branch3_2], 3)
    net = slim.batch_norm(net, is_training=True)
    net = slim.conv2d(net, 1024, 1, padding='SAME')
    # Global Average Pooling
    kernel_size = net.get_shape()[1:3]
    if kernel_size.is_fully_defined():
        net = slim.avg_pool2d(net, kernel_size, padding='VALID')
    else:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True)

    # dropout
    if dropout_keep_prob !=1:
        net = slim.dropout(net, dropout_keep_prob, is_training=True)

    # Use conv2d instead of fully_connected layers.
    net = slim.conv2d(net, 80, [1, 1])
    return net


def train():
    x_train, y_train = read_train_data()
    x_test, y_test = read_test_data()
    x_batch_train, y_batch_train = tf.train.shuffle_batch([x_train, y_train], batch_size=BATCH_SIZE, capacity=100,
                                                          min_after_dequeue=50, num_threads=3)
    x_batch_test, y_batch_test = tf.train.shuffle_batch([x_test, y_test], batch_size=BATCH_SIZE, capacity=100,
                                                        min_after_dequeue=50, num_threads=3)
    dropout = tf.placeholder('float')
    x = tf.placeholder(tf.float32, shape=[None, 229 * 229 * 3])
    y = tf.placeholder(tf.float32, shape=[None, 80])

    inputs = tf.reshape(x, shape=[BATCH_SIZE, 229, 229, 3])

    logits = net2(inputs, dropout)
    accuracy = get_accuracy(logits, y)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    global_step = tf.Variable(0, name='global_step')
    train_gdop = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy, global_step=global_step)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(2500):
        input_x, input_y = sess.run([x_batch_train, y_batch_test])
        input_x = np.reshape(input_x, [BATCH_SIZE, 229 * 229 * 3])
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: input_x, y: input_y, dropout: 1})
            loss = sess.run(cross_entropy, feed_dict={x: input_x, y: input_y, dropout: 1})
            print("Step: %d -----> accuracy: %g -----> loss: %g" % (i, acc, loss))
        sess.run(train_gdop, feed_dict={x: input_x, y: input_y, dropout: 0.7})
    save_model(sess, i)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    start = time.time()
    train()
    print("All steps use time: %g S" % (time.time() - start))
