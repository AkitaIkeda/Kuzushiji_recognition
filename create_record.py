import numpy as np
import pandas as pd
import cv2
import utils
import tensorflow as tf
import sys
import os
tf.compat.v1.enable_eager_execution()

OUT_PATH = 'train.tfrecords'
data_root = 'D:/datasets/kuzushiji-recognition/'
data_images = 'train_images/'
input_size = 128
input_shape = (input_size, input_size, 1)
class_num = 4787
data_num = 683464
batch_size = 4
du = pd.read_csv('D:/datasets/kuzushiji-recognition/unicode_translation.csv').reset_index()


def load_and_preprocess(image, label):
    image = image.numpy().astype(str)
    label = label.numpy().decode('UTF-8')
    x, y, w, h = int(image[1]), int(image[2]), int(image[3]), int(image[4])
    im = cv2.imread(image[0])
    im = im[y:y+h, x:x+w]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im =  cv2.threshold(im, cv2.threshold(im, 0, 255, 8)[0]+30, 255, cv2.THRESH_TRUNC)[1]
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = tf.compat.v1.image.resize_image_with_pad(im, input_size, input_size).numpy().astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.reshape(im, input_shape)
    mini = np.min(im)
    im = (im - mini)/(np.max(im) - mini)
    im = tf.cast(im, tf.float32)
    label = du.query('Unicode == "{}"'.format(label)).index[0]
    label = tf.cast(label, tf.uint32)
    return im, label

def wrapper(images, path):
    re = tf.py_function(load_and_preprocess, [images, path], (tf.float32, tf.uint32))
    re[0].set_shape(input_shape)
    re[1].set_shape([class_num])
    return re

def create():
    df = pd.read_csv('D:/datasets/kuzushiji-recognition/train.csv')
    images = []
    labels = []
    for index, item in df.iterrows():
        if type(item['labels']) == float:
            continue
        image_path = data_root+data_images+item['image_id']+'.jpg'
        l = np.array(item['labels'].split(' '))
        l = np.reshape(l, (-1, 5))
        for la, x, y, w, h in l:
            labels.append(la)
            images.append([image_path, x, y, w, h])
    return images, labels


def serialize(image, label):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy().tostring()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    return ex.SerializeToString()

def serialize_wrapper(image, label):
    out = tf.py_function(
        serialize,
        (image, label),
        tf.string)
    return tf.reshape(out, ())

ds = tf.data.Dataset.from_tensor_slices(create())
ds = ds.map(wrapper, num_parallel_calls=16)
ds = ds.map(serialize_wrapper, num_parallel_calls=16)
w = tf.data.experimental.TFRecordWriter(data_root+OUT_PATH)
w.write(ds)