from tensorflow.python.keras.optimizers import SGD
import models
import tensorflow as tf
from tensorflow.python import keras
import os
# import numpy as np
# import matplotlib.pyplot as plt
# tf.compat.v1.enable_eager_execution()

tfrecord = 'train.tfrecords'
input_size = 128
input_shape = (input_size, input_size, 1)
class_num = 4787
data_num = 683464
batch_size = 34


ds = tf.data.TFRecordDataset([tfrecord])
print(ds)


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


ds = ds.map(parse, num_parallel_calls=16)

# for i, l in ds.take(1):
#     plt.imshow(np.reshape(i.numpy(), (128, 128)))
#     plt.title(l)
#     plt.show()
#     exit()

ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size))
ds = ds.batch(batch_size).prefetch(1)

if os.path.isfile('models_10/model_data_0'):
    model = keras.models.load_model('./models/model_data_0')
else:
    model = models.build_wideresnet(
        input_shape, class_num, 8, [3, 4, 6, 3], 16)
    optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
model.summary()

for i in range(100):
    model.fit(ds, epochs=1, steps_per_epoch=data_num/batch_size)
    model.save('./models_10/model_data_'+str(i))
