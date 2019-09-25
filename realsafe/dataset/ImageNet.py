from __future__ import absolute_import

from realsafe.utils import get_dataset_path, show_progress

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import os, tarfile, six
from six.moves import urllib, range

def load_batches_imagenet_test(batch_size,
                               x_min,
                               x_max,
                               x_shape,
                               x_dtype,
                               y_dtype,
                               data_dir='ImageNet',
                               start=0,
                               end=50000,
                               label_offset=0,
                               return_target_class=False):
    
    # Download dataset if necessary
    dataset_path = get_dataset_path(data_dir)

    if not os.path.exists(os.path.join(dataset_path, 'ILSVRC2012_val_00000001.JPEG')):
        urllib.request.urlretrieve(
            'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar',
            os.path.join(dataset_path, 'ILSVRC2012_img_val.tar'), show_progress)
        tar = tarfile.open(os.path.join(dataset_path, 'ILSVRC2012_img_val.tar'), "r:tar")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, dataset_path)
        tar.close()

    # Read labels
    f = open(os.path.join(dataset_path, 'val.txt')).readlines()
    labels = {}
    for l in f:
        l = l.strip().split(' ')
        labels[l[0]] = int(l[1]) + label_offset

    # Read target class
    if return_target_class:
        f = open(os.path.join(dataset_path, 'target.txt')).readlines()
        target_labels = {}
        for l in f:
            l = l.strip().split(' ')
            target_labels[l[0]] = int(l[1]) + label_offset

    batch_shape = (batch_size,) + x_shape
    images = np.zeros(batch_shape, dtype=x_dtype.as_numpy_dtype)
    filenames = []
    idx = 0

    for filepath in sorted(tf.gfile.Glob(os.path.join(dataset_path, '*.JPEG')))[start:end]:
        img = imread(filepath, mode='RGB')

        height = img.shape[0]
        width = img.shape[1]
        center = int(0.875 * min(height, width))
        offset_height = (height - center + 1) // 2
        offset_width = (width - center + 1) // 2
        img = img[offset_height:offset_height+center, offset_width:offset_width+center,:]

        img = imresize(img, x_shape[:2], interp='bicubic').astype(x_dtype.as_numpy_dtype)
        img = img / 255.0 * (x_max - x_min) + x_min
        images[idx, :, :, :] = img
        filenames.append(os.path.basename(filepath))
        idx += 1

        if idx == batch_size:
            labels_for_batch = np.array([labels[n] for n in filenames]).astype(y_dtype.as_numpy_dtype)
            if return_target_class:
                target_labels_for_batch = np.array([target_labels[n] for n in filenames]).astype(y_dtype.as_numpy_dtype)
                yield filenames, images, labels_for_batch, target_labels_for_batch
            else:
                yield filenames, images, labels_for_batch
            filenames = []
            images = np.zeros(batch_shape, dtype=x_dtype.as_numpy_dtype)
            idx = 0
    if idx > 0:
        labels_for_batch = np.array([labels[n] for n in filenames]
            + [0] * (FLAGS.batch_size - len(filenames))).astype(y_dtype.as_numpy_dtype)
        if return_target_class:
            target_labels_for_batch = np.array([target_labels[n] for n in filenames]
                + [0] * (FLAGS.batch_size - len(filenames))).astype(y_dtype.as_numpy_dtype)
            yield filenames, images, labels_for_batch, target_labels_for_batch
        else:
            yield filenames, images, labels_for_batch


def load_image_of_class(label,
                        x_min,
                        x_max,
                        x_shape,
                        x_dtype,
                        y_dtype,
                        data_dir='ImageNet',
                        label_offset=0):

    dataset_path = get_dataset_path(data_dir)
    # Read labels
    f = open(os.path.join(dataset_path, 'val.txt')).readlines()
    labels = {}
    for l in f:
        l = l.strip().split(' ')
        labels[l[0]] = int(l[1]) + label_offset

    filenames = []
    for i in labels.keys():
        if labels[i] == label:
            filenames.append(i)

    for f in filenames:
        img = imread(os.path.join(dataset_path, f), mode='RGB')
        height = img.shape[0]
        width = img.shape[1]
        center = int(0.875 * min(height, width))
        offset_height = (height - center + 1) // 2
        offset_width = (width - center + 1) // 2
        img = img[offset_height:offset_height+center, offset_width:offset_width+center,:]

        img = imresize(img, x_shape[:2], interp='bicubic').astype(x_dtype.as_numpy_dtype)
        img = img / 255.0 * (x_max - x_min) + x_min
        yield img[np.newaxis]

