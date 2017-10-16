import pickle
import os
import cv2
import numpy as np
import tensorflow as tf


tf.flags.DEFINE_string('images', None, 'images directory')
tf.flags.DEFINE_string('masks', None, 'masks directory')
tf.flags.DEFINE_boolean('shuffle', False, 'shuffle the dataset')
tf.flags.DEFINE_string('output', None, 'output path')

FLAGS = tf.flags.FLAGS


def read_image(example):
  img = cv2.imread(os.path.join(FLAGS.images, example))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def read_mask(example):
  img = cv2.imread(os.path.join(FLAGS.masks, example))
  return img


def shuffle_dataset(images, masks):
  idx = np.random.permutation(len(images))
  images = images[idx, :, :, :]
  masks = masks[idx, :, :, :]
  return images, masks


def create_dataset():
  images = np.stack(map(read_image, sorted(os.listdir(FLAGS.images))), axis=0)
  masks = np.stack(map(read_mask, sorted(os.listdir(FLAGS.masks))), axis=0)
  masks = masks.clip(0, 1)
  if FLAGS.shuffle:
    images, masks = shuffle_dataset(images, masks)
  dataset = {'images': images, 'masks': masks}
  pickle.dump(file=open(FLAGS.output, 'wb'), obj=dataset)


if __name__ == '__main__':
    create_dataset()
