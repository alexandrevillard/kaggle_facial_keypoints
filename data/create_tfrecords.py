import argparse
import os

import numpy as np
import tensorflow as tf
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

IMG_SIZE = 96
FLIP_IDX = [
    (0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25),
]


def _bytes_feature(value):
    """
    Turns value to byte Feature
    
    :param value: 
    :return: Byte Feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """
    Turns value to int Feature
    
    :param value: 
    :return: int Feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """
    Turns value to Feature
    
    :param value: 
    :return: float Feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_rec(img, labels, name='train'):
    """
    Creates TFRecords files given arrays of img and corresponding labels.
    :param img: nparray of flattened images
    :param labels: nparray of labels
    :param name: name of the output .tfrecords file
    :return: None
    """
    writer = tf.python_io.TFRecordWriter('{}.tfrecords'.format(name))
    for u, v in zip(img, labels):
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMG_SIZE),
            'width': _int64_feature(IMG_SIZE),
            'image_raw': _bytes_feature(u.tostring()),
            'labels': _float_feature(v)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def split_data(*arrays, **opts):
    """
    Split arrays, e.g. for train/valid data sets.
    If x, y are such that x.shape = (L, m), y.shape = (L, n), returned value is
    x_1, x_2, y_1, y_2 where 
        x_1.shape = (L * ratio, m)
        x_2.shape = (L * (1 - ratio), m)
        y_1.shape = (L * ratio, n)
        y_2.shape = (L * (1 - ratio, n)
    :param arrays: arrays to split
    :param opts: options, e.g.
        ratio: train/total ratio
        random_state: seed
    :return: the splitted array
    """
    n = int(len(arrays[0]) * opts.get('ratio', 0.75))
    res = shuffle(*arrays, random_state=opts.get('random_state', 42))
    f_res = []
    for r in res:
        f_res.append(r[:n])
        f_res.append(r[n:])
    return f_res


def flip(img):
    """
    Flips a flattened image
    :param img: 
    :return: 
    """
    img = np.reshape(img, (IMG_SIZE, IMG_SIZE))
    img = np.fliplr(img)
    return img.flatten()


def run(
        filename,
):
    path = os.path.expanduser(filename)
    df = read_csv(path)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df = df.dropna()
    imgs = np.vstack(df['Image'].values)
    imgs = imgs.astype(np.uint8)
    labels = df[df.columns[1:-1]].values
    labels = (labels - 48) / 48
    labels = labels.astype(np.float32)
    imgs_train, img_pred, labels_train, labels_pred = split_data(imgs, labels)
    imgs_train_aug, labels_train_aug = augment(imgs_train, labels_train)
    create_tf_rec(imgs_train, labels_train, 'train')
    create_tf_rec(imgs_train_aug, labels_train_aug, 'train-aug')
    create_tf_rec(img_pred, labels_pred, 'pred')


def augment(imgs, labels):
    """
    Augments the data set by flipping the image horizontally.
    :param imgs: 
    :param labels: 
    :return: a tuple containing the augmented images and labels datasets.
    """
    imgs_aug = np.apply_along_axis(flip, axis=1, arr=imgs)
    imgs_aug = np.concatenate([imgs, imgs_aug], axis=0)
    labels_aug = labels.copy()
    labels_aug[:, ::2] = labels_aug[:, ::2] * (-1)  # flip x components
    for (x, y) in FLIP_IDX:  # swap right and left elements, e.g. eyes
        tmp = labels_aug[:, y].copy()
        labels_aug[:, y] = labels_aug[:, x].copy()
        labels_aug[:, x] = tmp
    labels_aug = np.concatenate([labels, labels_aug], axis=0)
    return imgs_aug, labels_aug

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        required=True,
                        type=str,
                        help='Absolute path of the csv file.')
    parser.add_argument('--aug',
                        default=False,
                        type=bool,
                        help='Flag set to True if data augmented (flipped horizontally).')
