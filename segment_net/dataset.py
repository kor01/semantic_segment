import os
import tensorflow as tf
from tensorflow.contrib import data as dt

tf.flags.DEFINE_string('train', None, 'train images and masks directory')
tf.flags.DEFINE_string('valid', None, 'validate images and masks directory')
tf.flags.DEFINE_boolean('normalize', True, 'normalize image')
tf.flags.DEFINE_integer('threads', 4, 'number of preprocess threads')
tf.flags.DEFINE_integer('crop_size', 64, 'crop size')
tf.flags.DEFINE_integer('batch_size', 64, 'number pf filters')

FLAGS = tf.flags.FLAGS


def list_files(images_path, masks_path, shuffle):
  images = sorted(os.listdir(images_path))
  masks = sorted(os.listdir(masks_path))
  images = list(map(lambda x: os.path.join(images_path, x), images))
  masks = list(map(lambda x: os.path.join(masks_path, x), masks))
  images, masks = map(tf.constant, (images, masks))
  dataset = dt.Dataset.from_tensor_slices([images, masks])
  assert isinstance(dataset, dt.Dataset)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=tf.to_int64(tf.shape(images)[0]))
  return dataset


def read_images(dataset, width, height, classes):

  def read_fn(image, mask):
    image, mask = map(tf.image.decode_jpeg,
                      map(tf.read_file, (image, mask)))
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


def create_dataset(width, height, classes,
                   is_train=True, batch_size=None):
  path = FLAGS.train if is_train else FLAGS.valid
  images, masks = map(lambda x: os.path.join(path, x), ('images', 'masks'))
  dataset = list_files(images, masks, shuffle=is_train)
  dataset = read_images(dataset, width, height, classes)
  if is_train:
    dataset = random_crop_images(dataset)
  assert isinstance(dataset, dt.Dataset)
  batch_size = batch_size or FLAGS.batch_size
  dataset = dataset.batch(batch_size)
  return dataset
