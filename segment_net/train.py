import atexit
import sys
import os
import datetime
from segment_net.flags import *
from segment_net.network import build_network
from segment_net.dataset import create_dataset
from segment_net.dataset import glob_list_dir
from segment_net.dataset import list_files


LABEL_WEIGHTS = [1, 1, 1]


def timestamp():
  return str(datetime.datetime.now().ctime())


def create_log():
  if FLAGS.log is None:
    log_file = sys.stderr
  else:
    log_file = open(FLAGS.log, 'w')

  def _close():
    if log_file != sys.stderr:
      log_file.close()
  atexit.register(_close)

  def _log_info(*args):
    print(*args, file=log_file)
    log_file.flush()
  return _log_info

log_info = create_log()


def label_weights():
  weights = tf.constant(LABEL_WEIGHTS, dtype=tf.float32)
  return weights / tf.reduce_sum(weights)


def validate_network(sess):

  with tf.name_scope('validation'):
    images, masks = list_files(FLAGS.valid)
    dataset = create_dataset(
      image_files=images, mask_files=masks, batch_size=32)
    iterator = dataset.make_initializable_iterator()
    images, masks = iterator.get_next()
    tensors = build_network(
      name=FLAGS.name, reuse=True, train=False,
      images=images, masks=masks, iou=True)

  def validate_fn():
    sess.run(iterator.initializer)
    while True:
      try:
        sess.run(tensors.update_iou)
      except tf.errors.OutOfRangeError:
        break
    iou_val = sess.run(tensors.mean_iou)
    return iou_val

  return validate_fn


def zero_out_confusion_matrix(sess):
  local_vars = tf.local_variables()
  cms = filter(lambda x: 'confusion' in x.name, local_vars)
  zeros_ops = [x.assign(tf.zeros_like(x)) for x in cms]
  op = tf.group(*zeros_ops)

  def zero_cms():
    print('running ops:', zeros_ops)
    sess.run(op)
    print('done')
  return zero_cms


def train_network(sess):
  images, masks = list_files(FLAGS.train)
  dataset = create_dataset(
    image_files=images, mask_files=masks, is_train=True)
  iterator = dataset.make_initializable_iterator()
  images, masks = iterator.get_next()

  with tf.name_scope('train'):
    tensors = build_network(
      FLAGS.name, images=images,
      masks=masks, train=True, iou=True)

  def train_fun(epoch):
    total_loss, counter = 0, 0
    sess.run(iterator.initializer)
    while True:
      try:
        loss, _, _ = sess.run(
          [tensors.loss, tensors.train_op,
           tensors.update_iou])
        total_loss += loss
        counter += 1
        if counter % 10 == 0:
          train_iou = sess.run(tensors.mean_iou)
          log_info('[%s] %d samples processed, loss [%f] iou [%f]'
                   % (timestamp(), counter * FLAGS.batch_size,
                      total_loss / counter, train_iou))
      except tf.errors.OutOfRangeError:
        break
    train_iou = sess.run(tensors.mean_iou)
    average_loss = total_loss / counter
    log_info('[%s] epoch [%d] training completed'
             % (timestamp(), epoch))
    return train_iou, average_loss
  return train_fun


def score_dataset(name):
  path = os.path.join(FLAGS.evaluate, name)
  images, masks = list_files(path)
  dataset = create_dataset(
    image_files=images, mask_files=masks,
    is_train=False, batch_size=32)
  return dataset


def score_datasets():
  follow_dataset = score_dataset('following_images')
  target_dataset = score_dataset('patrol_with_targ')
  image_files = os.path.join(FLAGS.evaluate, '*/images/*.jpeg')
  mask_files = os.path.join(FLAGS.evaluate, '*/masks/*.png')
  image_files, mask_files = glob_list_dir(image_files, mask_files)
  concat_dataset = create_dataset(
    image_files, mask_files, batch_size=32, mask_fmt='png')

  return follow_dataset, target_dataset, concat_dataset


def sample_from_dataset(dataset):
  config = tf.ConfigProto(device_count={'GPU': 0})
  sess = tf.Session(config=config)
  with sess.as_default():
    dataset = score_dataset(dataset)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

  def sample_fn():
    return sess.run(batch)

  return sample_fn


def score_networks(sess):
  with tf.name_scope('score'):
    follow_dataset = score_dataset('following_images')
    target_dataset = score_dataset('patrol_with_targ')
    image_files = os.path.join(FLAGS.evaluate, '*/images/*.jpeg')
    mask_files = os.path.join(FLAGS.evaluate, '*/masks/*.png')
    image_files, mask_files = glob_list_dir(image_files, mask_files)
    concat_dataset = create_dataset(
      image_files, mask_files, batch_size=32, mask_fmt='png')

  follow_fn = score_network('follow', sess, follow_dataset)
  target_fn = score_network('target', sess, target_dataset)
  concat_fn = score_network('concat', sess, concat_dataset)

  def score_fn():
    target_iou = target_fn()
    follow_iou = follow_fn()
    print('iou:', follow_iou, target_iou)
    final_iou = (follow_fn() + target_fn()) / 2
    weight = concat_fn()
    return final_iou, weight

  return score_fn


def score_network(name, sess, dataset):
  with tf.name_scope(name):
    iterator = dataset.make_initializable_iterator()
    images, masks = iterator.get_next()
    tensors = build_network(
      name=FLAGS.name, images=images,
      masks=masks, train=False, reuse=True, iou=True)

  def score_fn():
    sess.run(iterator.initializer)
    while True:
      try:
        sess.run(tensors.update_iou)
      except tf.errors.OutOfRangeError:
        break
    iou_val = sess.run(tensors.mean_iou)
    return iou_val

  return score_fn


def get_cms(name):
  local_vars = tf.local_variables()
  cms = filter(lambda x: 'confusion' in x.name, local_vars)
  cms = list(filter(lambda x: name in x.name, cms))
  assert len(cms) == 1, 'not found or not unique %d' % len(cms)
  return cms[0]


def save_network(sess):
  saver = tf.train.Saver(
    var_list=tf.trainable_variables())
  path = os.path.join(FLAGS.save, 'segment')

  def restore_fn():
    ckpt = tf.train.get_checkpoint_state(FLAGS.save)
    if ckpt is not None:
      log_info('[%s] restore from previous ckpt [%s]'
               % (timestamp(), ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)

  def save_fn(epoch):
    saver.save(sess, path, global_step=epoch)
  return save_fn, restore_fn


def train_main():
  sess = tf.Session()
  train_fn = train_network(sess)
  validate_fn = validate_network(sess)
  score_fn = score_networks(sess)
  zero_cms_fn = zero_out_confusion_matrix(sess)
  save_fn, restore_fn = save_network(sess)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  restore_fn()
  for e in range(FLAGS.epoch):
    train_iou, average_loss = train_fn(e)

    if (e + 1) % FLAGS.evaluate_step == 0:
      validate_iou = validate_fn()
      final_iou, weight = score_fn()
      score = final_iou * weight
      log_info('[%s] epoch %d average_loss [%f] '
               'train_iou [%f] validate_iou [%f] '
               'final_iou [%f] weight [%f] final_score [%f]'
               % (timestamp(), e, average_loss,
                  train_iou, validate_iou, final_iou, weight, score))
    else:
      log_info('[%s] epoch %d average_loss [%f] train_iou [%f]'
               % (timestamp(), e, average_loss, train_iou))

    save_fn(e)
    zero_cms_fn()
