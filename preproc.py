import cv2
import os
from multiprocessing import Pool
import numpy as np
import tensorflow as tf

TEST_PATH = 'D:/datasets/kuzushiji-recognition/test_images/'


def pre(f):
    if os.path.isfile(os.path.join('dataset/validate', f)):
        return
    image = cv2.cvtColor(cv2.imread(
        TEST_PATH + f), cv2.COLOR_BGR2RGB)
    image = cv2.addWeighted(
        image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
    cv2.imwrite(os.path.join('dataset/validate',
                             f), image)
    print('{} done'.format(f))


if __name__ == "__main__":
    with Pool(4) as p:
        p.map(pre, os.listdir(TEST_PATH))

# ds = tf.data.Dataset.from_tensor_slices(
#     tf.reshape(os.listdir(TEST_PATH), (-1, 1)))
# print(ds)
# ds = ds.map(lambda x: tf.py_function(pre, [x], tf.string))
# for i, im in enumerate(ds.take(-1)):
#     print('{}: {}'.format(i, im.numpy()[0].decode()))
