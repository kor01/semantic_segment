import os
import glob
from tensorflow.contrib import data as dt
from segment_net.flags import *


def list_files(path):
  images_path, masks_path = \
    map(lambda x: os.path.join(path, x), ('images', 'masks'))
  images = sorted(os.listdir(images_path))
  masks = sorted(os.listdir(masks_path))
  images = list(map(lambda x: os.path.join(images_path, x), images))
  masks = list(map(lambda x: os.path.join(masks_path, x), masks))
  images, masks = map(tf.constant, (images, masks))
  dataset = dt.Dataset.from_tensor_slices([images, masks])
  assert isinstance(dataset, dt.Dataset)
  return images, masks


def list_multiple_dirs(images_paths, masks_path):
  images = map(lambda x: sorted(os.listdir(x)), images_paths)
  masks = map(lambda x: sorted(os.listdir(x)), masks_path)

  def add_prefix(prefix_files):
    prefix, files = prefix_files
    return list(map(lambda x: os.path.join(prefix, x), files))

  images = sum(map(zip(images_paths, images), add_prefix), [])
  masks = sum(map(zip(masks_path, masks), add_prefix), [])
  images, masks = map(tf.constant, (images, masks))
  return images, masks


def glob_list_dir(images, masks):
  print(images, masks)
  images = sorted(glob.glob(images))
  masks = sorted(glob.glob(masks))
  images, masks = map(tf.constant, (images, masks))
  return images, masks


def filename_dataset(images, masks, shuffle):
  dataset = dt.Dataset.from_tensor_slices([images, masks])
  if shuffle:
    dataset = dataset.shuffle(buffer_size=tf.to_int64(tf.shape(images)[0]))
  return dataset


def read_images(dataset, width, height, classes,
                image_fmt, mask_fmt):

  def read_fn(image, mask):
    image = tf.read_file(image)
    mask = tf.read_file(mask)
    image_decoder = tf.image.decode_jpeg \
      if image_fmt == 'jpeg' else tf.image.decode_png
    mask_decoder = tf.image.decode_jpeg \
      if mask_fmt == 'jpeg' else tf.image.decode_png
    image = image_decoder(image)
    mask = mask_decoder(mask)
    image, mask = map(tf.to_float, (image, mask))
    assert isinstance(image, tf.Tensor)
    image.set_shape((width, height, 3))
    mask.set_shape((width, height, classes))
    mask = tf.clip_by_value(mask, 0, 1)
    return image, mask

  def normalize_fn(image, mask):
    if not FLAGS.normalize:
      return image, mask
    image = (image / 255. - 0.5) * 2
    return image, mask

  assert isinstance(dataset, dt.Dataset)
  return dataset.map(read_fn).map(normalize_fn)


def random_crop_images(dataset):

  def crop_fn(image, mask):
    if FLAGS.crop_size <= 0:
      return image, mask
    concat = tf.concat((image, mask), axis=-1)
    concat = tf.random_crop(
      concat, size=(FLAGS.crop_size, FLAGS.crop_size, 6))
    image, mask = concat[:, :, :3], concat[:, :, 3:]
    return image, mask
  assert isinstance(dataset, dt.Dataset)
  return dataset.map(crop_fn)


def create_dataset(image_files, mask_files,
                   width=None, height=None, classes=None,
                   is_train=False, batch_size=None,
                   image_fmt='jpeg', mask_fmt='jpeg'):
  width = width or FLAGS.width
  height = height or FLAGS.height
  classes = classes or FLAGS.classes
  dataset = filename_dataset(
    image_files, mask_files, shuffle=is_train)
  dataset = read_images(dataset, width, height,
                        classes, image_fmt, mask_fmt)
  if is_train:
    dataset = random_crop_images(dataset)

  assert isinstance(dataset, dt.Dataset)
  batch_size = batch_size or FLAGS.batch_size
  dataset = dataset.batch(batch_size)
  return dataset
