#!/usr/bin/env python
# _*_coding:utf-8_*_
import tensorflow as tf
import random
import os
import cv2
import pandas as pd


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def creatTrainData(HOME_PATH):
    json = pd.read_json(os.path.join(HOME_PATH, 'scene_train_annotations.json'))
    image_id = json['image_id']
    label_id = json['label_id']
    filenameTrain = 'TFRecord/train.tfrecords'
    writerTrain = tf.python_io.TFRecordWriter(filenameTrain)
    i = 0
    image_folder = os.path.join(HOME_PATH, 'scene_train_images')
    for name in image_id:
        path = os.path.join(image_folder, name)
        raw = cv2.imread(path)
        res = cv2.resize(raw, (229, 229))
        data = res.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label_id[i]),
            'image_raw': _bytes_feature(data)
        }))
        writerTrain.write(example.SerializeToString())
        i += 1
    writerTrain.close()


def creatTestData(HOME_PATH):
    json = pd.read_json(os.path.join(HOME_PATH, 'scene_validation_annotations.json'))
    image_id = json['image_id']
    label_id = json['label_id']
    filenameTest = 'TFRecord/test.tfrecords'
    writerTest = tf.python_io.TFRecordWriter(filenameTest)
    i = 0
    image_folder = os.path.join(HOME_PATH, 'scene_validation_images')
    for name in image_id:
        path = os.path.join(image_folder, name)
        raw = cv2.imread(path)
        res = cv2.resize(raw, (229, 229))
        data = res.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label_id[i]),
            'image_raw': _bytes_feature(data)
        }))
        writerTest.write(example.SerializeToString())
        i += 1
    writerTest.close()

if __name__ == '__main__':
    creatTestData('data/scene_validation/')
    print('test done')
    creatTrainData('data/scene_train/')
    print('train done')

