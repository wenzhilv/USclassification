
import glob
import os

import numpy as np
import matplotlib.image as mptim
import matplotlib.pyplot as mplt
import skimage.io
import skimage.transform
import skimage.color

def img_rgb2gray(img):
    pass

def img_read(path):
    resimg = skimage.io.imread(path)


def img_read_resize_gray(path,image_width,image_height):
    img = skimage.io.imread(path)
    imgsresize = skimage.transform.resize(img, (image_width, image_height))
    imgray = skimage.color.rgb2gray(imgsresize)
    return imgray


def test_imread(dir, num_classes, image_width, image_height, imgextension='png', img_depth = 3):
    """

    :param dir:
    :param num_classes:
    :param image_width:
    :param image_height:
    :param imgextension:
    :param img_depth:
    :return:
    """
    clasindex = []
    cntimg = 0
    for index in range(num_classes):
        filepath = os.path.join(dir, str(index), '*.' + imgextension)
        labelfiles = glob.glob(filepath)
        clasindex.append(len(labelfiles))
        cntimg += len(labelfiles)
    resimg = np.empty([cntimg, image_width,image_height], dtype=np.uint8)
    reslabel = np.zeros([cntimg, 1], dtype=np.uint8)

    cntimg = 0
    for index in range(num_classes):
        filepath = os.path.join(dir, str(index), '*.'+imgextension)
        labelfiles = glob.glob(filepath)
        for file in labelfiles:
            img_gray = img_read_resize_gray(file, image_width, image_height)
            resimg[cntimg, :, :] = img_gray
            reslabel[cntimg, 0] = index
            cntimg += 1
    cntimg = 0
    return resimg, reslabel


def test_imread2(dir, num_classes, image_width, image_height, imgextension='png', img_depth=1):
    """

    :param dir:
    :param num_classes:
    :param image_width:
    :param image_height:
    :param imgextension:
    :param img_depth:
    :return:resimg -- [imgnum, size, size, depth]
            reslabel2 -- [imgnum, num_classes]
    """
    clasindex = []
    cntimg = 0
    for index in range(num_classes):
        filepath = os.path.join(dir, str(index), '*.' + imgextension)
        labelfiles = glob.glob(filepath)
        clasindex.append(len(labelfiles))
        cntimg += len(labelfiles)
    resimg = np.empty([cntimg, image_width,image_height], dtype=np.uint8)
    reslabel = np.zeros([cntimg, 1], dtype=np.uint8)

    cntimg = 0
    for index in range(num_classes):
        filepath = os.path.join(dir, str(index), '*.'+imgextension)
        labelfiles = glob.glob(filepath)
        for file in labelfiles:
            img_gray = img_read_resize_gray(file, image_width, image_height)
            resimg[cntimg, :, :] = img_gray
            reslabel[cntimg, 0] = index
            cntimg += 1
    cntimg = 0
    reslabel2 = np.zeros([cntimg, num_classes], dtype=np.uint8)
    for index in range(cntimg):
        if 0 == reslabel[index]:
            reslabel2[index, 0] = 1
        elif 1 == reslabel[index]:
            reslabel2[index, 1] = 1
        else:
            raise ValueError('Not supported class %d data', index)

    return resimg, reslabel2



def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = skimage.io.imread(im)
            img = skimage.transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


if __name__ == '__main__':
    print('run main start')
    dirs = r'D:\lwz\softproject\data\USimg\cui'
    pass
    num_classes = 2
    test_imread(dirs, num_classes, 256, 256, 'png')
    print('run main over')
