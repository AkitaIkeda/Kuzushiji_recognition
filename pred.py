import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from tensorflow.python import keras


TEST_PATH = 'D:/datasets/kuzushiji-recognition/test_images/'
CSV_OUT = './out.csv'
MODEL_DATA = './model.ckpt'
input_size = 128
input_shape = (input_size, input_size, 1)
KERAS_MODEL = './models/model_data_6'
batch_size = 48


def ocr_network(input, is_training):
    conv1 = conv2d(input, 64, 3, 1)  # [N, 728, 448, 64]
    conv2 = conv2d(conv1, 64, 3, 1)
    pool1 = tf.layers.max_pooling2d(
        conv2, pool_size=2, strides=2)  # [N, 364, 224, 64]

    conv3 = conv2d(pool1, 64, 3, 1)  # [N, 364, 224, 64]
    dropout1 = tf.layers.dropout(pool1, rate=0.25, training=is_training)

    conv4 = conv2d(conv3, 64, 3, 1)
    conv5 = conv2d(dropout1, 128, 12, 14)  # [N, 26, 16, 128]

    pool2 = tf.layers.max_pooling2d(
        conv4, pool_size=2, strides=2)  # [N, 182, 112, 64]
    dropout2 = tf.layers.dropout(
        conv5, rate=0.5, training=is_training)  # [N, 26, 16, 128]

    dropout3 = tf.layers.dropout(pool2, rate=0.25, training=is_training)
    dense1 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)
    dense3 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)

    conv6 = conv2d(dropout3, 128, 12, 14)  # [N, 13, 8, 128]
    dense4 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu)
    dense5 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu)
    dense6 = tf.layers.dense(dense3, units=64, activation=tf.nn.relu)

    dropout4 = tf.layers.dropout(conv6, rate=0.5, training=is_training)
    dense7 = tf.layers.dense(dense4, units=1)
    dense8 = tf.layers.dense(dense5, units=2, activation=tf.nn.sigmoid)
    dense9 = tf.layers.dense(dense6, units=2, activation=tf.nn.sigmoid)

    concat1 = tf.concat([dense7, dense8, dense9], axis=-1)  # [N, 26, 16, 5]

    dense10 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense11 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense12 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)

    dense13 = tf.layers.dense(dense10, units=64, activation=tf.nn.relu)
    dense14 = tf.layers.dense(dense11, units=64, activation=tf.nn.relu)
    dense15 = tf.layers.dense(dense12, units=64, activation=tf.nn.relu)

    dense16 = tf.layers.dense(dense13, units=1)
    dense17 = tf.layers.dense(dense14, units=2, activation=tf.nn.sigmoid)
    dense18 = tf.layers.dense(dense15, units=2, activation=tf.nn.sigmoid)

    concat2 = tf.concat([dense16, dense17, dense18], axis=-1)  # [N, 13, 8, 5]

    return concat1, concat2


def conv2d(input, filters, kernel_size, strides, padding='same'):
    conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            activation=tf.nn.relu)
    return conv


def reorg(feature_map):
    grid_size = tf.shape(feature_map)[1:3]
    image_size = tf.cast([image_h, image_w], tf.int32)
    ratio = tf.cast(image_size / grid_size, tf.float32)

    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    xy_offset = tf.concat([x_offset, y_offset], axis=-1)
    xy_offset = tf.cast(tf.reshape(
        xy_offset, shape=[grid_size[0], grid_size[1], 2]), tf.float32)

    conf_logits, box_centers, box_sizes = tf.split(
        feature_map, [1, 2, 2], axis=-1)

    box_centers = box_centers + xy_offset
    box_centers = box_centers * ratio[::-1]

    box_sizes = tf.exp(box_sizes) * ratio[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)

    return xy_offset, boxes, conf_logits


def predict(feature_maps):
    results = [reorg(feature_map) for feature_map in feature_maps]

    def _reshape(result):
        xy_offset, boxes, conf_logits = result
        grid_size = tf.shape(xy_offset)[:2]
        boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1], 4])
        conf_logits = tf.reshape(
            conf_logits, [-1, grid_size[0] * grid_size[1], 1])
        return boxes, conf_logits

    boxes_list, confs_list = [], []
    for result in results:
        boxes, conf_logits = _reshape(result)
        confs = tf.nn.sigmoid(conf_logits)
        boxes_list.append(boxes)
        confs_list.append(confs)

    boxes = tf.concat(boxes_list, axis=1)
    confs = tf.concat(confs_list, axis=1)

    center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    xmin = center_x - width / 2
    xmax = center_x + width / 2
    ymin = center_y - height / 2
    ymax = center_y + height / 2

    boxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    return boxes, confs


def get_image(line):
    image = cv2.cvtColor(cv2.imread(
        TEST_PATH+line), cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (image_w, image_h))
    image = np.asarray(image, np.float32)
    image = image / 255.
    return image


def create(f, boxes):
    image = []
    for box in boxes:
        image.append([TEST_PATH+f, box[0], box[1],
                      box[2]-box[0], box[3]-box[2]])
    return image


def load_and_preprocess(image):
    image = image.numpy().astype(str)
    x, y, w, h = int(image[1]), int(image[2]), int(image[3]), int(image[4])
    im = cv2.imread(image[0])
    im = im[y:y+h, x:x+w]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(im, cv2.threshold(im, 0, 255, 8)
                       [0]+30, 255, cv2.THRESH_TRUNC)[1]
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = tf.compat.v1.image.resize_image_with_pad(
        im, input_size, input_size).numpy().astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.reshape(im, input_shape)
    mini = np.min(im)
    im = (im - mini)/(np.max(im) - mini)
    im = tf.cast(im, tf.float32)
    return im


def wrapper(images, path):
    re = tf.py_function(load_and_preprocess,
                        images, tf.float32)
    re.set_shape(input_shape)
    return re


is_training = tf.placeholder_with_default(
    False, shape=None, name='is_training')

images = os.listdir(TEST_PATH)

config = tf.ConfigProto()
config.allow_soft_placement
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
df = pd.DataFrame(columns=['image_id', 'labels'])
du = pd.read_csv('D:/datasets/kuzushiji-recognition/unicode_translation.csv')
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_DATA)
    for f in images:
        y_pred_l, y_pred_s = ocr_network(get_image(f), is_training=is_training)
        pred_boxes, pred_confs = predict([y_pred_l, y_pred_s])
        pred_boxes = (pred_boxes.eval(session=sess))
        pred_confs = (pred_confs.eval(session=sess))
        boxes = pred_boxes[pred_confs > 0.85]

        ds = tf.data.Dataset.from_tensor_slices(
            create(f, boxes))
        ds = ds.map(wrapper)
        model = keras.models.load_model(KERAS_MODEL)
        out = model.predict(ds, batch_size=batch_size)
        re = []
        for i, l in enumerate(out):
            re.append(
                ' '.join([du['Unicode'][np.max(l)], boxes[i][0], boxes[i][1]]))
        df.append([f.split('.')[0], ' '.join(re)])
