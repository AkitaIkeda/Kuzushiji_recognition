from tensorflow.python import keras
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# tf.compat.v1.enable_eager_execution()

tfrecord = 'train.tfrecords'
input_size = 128
input_shape = (input_size, input_size, 1)
class_num = 4787
data_num = 683464
batch_size = 48
du = pd.read_csv(
    'D:/datasets/kuzushiji-recognition/unicode_translation.csv')


def parse(ex):
    out = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    out = tf.io.parse_single_example(ex, out)
    image = tf.decode_raw(out['image'], tf.float32)
    label = tf.one_hot(out['label'], class_num)
    label = tf.cast(label, tf.uint8)
    image = tf.reshape(image, input_shape)
    return image, label


ds = tf.data.TFRecordDataset([tfrecord])
ds = ds.map(parse)
ds = ds.batch(batch_size)

# model = keras.models.load_model('./models/model_data_5')

for model_path in os.listdir('./models/'):
    model = keras.models.load_model('./models/'+model_path)
    out = model.evaluate(ds, steps=128)
#     classes_list = model.predict(ds, steps=5)
#     out = '{}: {}\n'.format(model_path, len(classes_list))
#     for classes in classes_list:
#         for i in np.array(range(4787))[classes > 0.3]:
#             out += '\t{}: {}\n'.format(du['char'][i], classes[i])
#         out += '\n'
#     print(out)

# for i, image in enumerate(ds.take(5)):
#     plt.subplot(1, 5, i+1)
#     plt.imshow(np.reshape(image.numpy(), (128, 128)))
#     plt.title(('dataset.{}'.format(i)))
# plt.show()
