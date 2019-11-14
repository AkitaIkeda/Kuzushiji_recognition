import cv2
from functools import reduce
import os
import numpy as np
import matplotlib.pyplot as plt

def compose(*funcs):
    '''複数の層を結合する。
    '''
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def preprocessing(image):
    '''画像の前処理

    Arguments:
        image: 処理する画像
    '''
    added = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(added, cv2.threshold(added, 0, 255, cv2.THRESH_OTSU)[0]+30, 255, cv2.THRESH_TRUNC)[1]