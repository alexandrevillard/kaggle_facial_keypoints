import numpy as np
import tensorflow as tf
import os
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle


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
            'height': _int64_feature(height),
            'width': _int64_feature(width),
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


fle = 'output_dir'
df = read_csv(os.path.expanduser(fle))
df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
df = df.dropna()
img = np.vstack(df['Image'].values)
img = img.astype(np.uint8)

labels = df[df.columns[:-1]].values
labels = (labels - 48) / 48
labels = labels.astype(np.float32)

img_train, img_pred, labels_train, labels_pred = split_data(img, labels)
create_tf_rec(img_train, labels_train, 'train')
create_tf_rec(img_pred, labels_pred, 'pred')
