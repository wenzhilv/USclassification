# one-hot-encoding
"""

"""

import numpy as np


def one_hot_encoding(label, class_num):
    """
    :param label:   [num, 1]
    :param class_num:
    :return: [num, class_num]
    """
    res = []
    tmpindex = np.zeros([class_num, class_num], dtype=np.uint16)
    labelindex = np.zeros([len(label), class_num], dtype=np.uint16)
    for index in range(class_num):
        tmpindex[index, index] = 1
    for idx, folder in enumerate(label):
        labelindex[idex, :] = tmpindex[label[idx], :]
    res = labelindex
    return res
