import atexit
import sys
import os
import datetime
import tensorflow as tf
from collections import namedtuple
from segment_net.dataset import create_dataset
from segment_net.baseline import fcn_model


LABEL_WEIGHTS = [1, 1, 1]


tf.flags.DEFINE_string('log', None, 'log file, None to stderr')
tf.flags.DEFINE_string('save', None, 'model ckpt path')
tf.flags.DEFINE_integer('epoch', 20, 'number of epoch')
tf.flags.DEFINE_integer('classes', 3, 'number of semantic classes')
tf.flags.DEFINE_float('lr', 5e-2, 'learning rate')
tf.flags.DEFINE_string('name', 'fcn_model', 'name of network')


IMAGE_SHAPE = (256, 256)
FLAGS = tf.flags.FLAGS
NETWORKS = {'fcn_model': fcn_model}

OutputTensors = namedtuple(
  'OutputTensors', ('images', 'masks', 'logits', 'loss',
                    'train_op', 'mean_iou', 'update_iou'))


def get_network(name):
  assert name in NETWORKS, 'unimplemented %s' % name
  return NETWORKS[name]


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


def stable_log(tensor):
  ret = tf.log(tensor + 1e-5)
  return ret


def crop_images(image, mask):
  concat = tf.concat([image, tf.to_float(mask)], axis=-1)
  print('concat:', concat)
  patches = [tf.random_crop(
    concat, size=(FLAGS.batch_size, 32, 32, 6))
    for _ in range(FLAGS.num_patches)]
  batch = tf.concat(patches, axis=0)
  return batch[:, :, :, :3], batch[:, :, :, 3:]


def label_weights():
  weights = tf.constant(LABEL_WEIGHTS, dtype=tf.float32)
  return weights / tf.reduce_sum(weights)


def segment_loss(logits, classes):
  sm = tf.nn.softmax(logits, dim=-1)
  # xent = -stable_log(sm) * classes
  xent = -tf.log(sm) * classes
  # xent = xent * label_weights()
  loss = tf.reduce_sum(xent, axis=-1)
  loss = tf.reduce_mean(loss)
  return loss


def measure_iou(logits, masks):
  labels = tf.argmax(masks, axis=-1)
  pred = tf.argmax(logits, axis=-1)
  mean, update = tf.metrics.mean_iou(
    predictions=pred, labels=labels, num_classes=3)
  return mean, update


def build_network(
    name, images=None,
    masks=None, train=False, reuse=None, iou=False):

  network = get_network(name)
  with tf.variable_scope(name, reuse=reuse):

    if images is None:
      shape = (None,) + IMAGE_SHAPE + (3,)
      images = tf.placeholder(
        dtype='float32', shape=shape)
    if masks is None and (iou or train):
      shape = (None,) + IMAGE_SHAPE + (FLAGS.classes,)
      masks = tf.placeholder(
        dtype='float32', shape=shape)

    logits = network(images, FLAGS.classes)

    if iou:
      mean_iou, update_iou = measure_iou(logits, masks)
    else:
      mean_iou, update_iou = None, None

    if train:
      loss = segment_loss(logits, tf.to_float(masks))
      train_op = tf.train.AdamOptimizer(
        learning_rate=FLAGS.lr).minimize(loss)
    else:
      loss, train_op = None, None

    ret = OutputTensors(
      images=images, masks=masks, logits=logits, loss=loss,
      train_op=train_op, mean_iou=mean_iou, update_iou=update_iou)
    return ret


def validate_network(sess):

  with tf.name_scope('validation'):
    dataset = create_dataset(
      is_train=False, batch_size=256, width=IMAGE_SHAPE[0],
      height=IMAGE_SHAPE[1], classes=FLAGS.classes)
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
  with tf.control_dependencies(
      list(map(lambda x: x.assign_sub(x), cms))):
    op = tf.no_op()

  def zero_cms():
    sess.run(op)
  return zero_cms


def train_network(sess):
  dataset = create_dataset(
    is_train=True, width=IMAGE_SHAPE[0],
    height=IMAGE_SHAPE[1], classes=FLAGS.classes)
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


def save_network(sess):
  saver = tf.train.Saver(
    var_list=tf.trainable_variables())
  path = os.path.join(FLAGS.save, 'segment')

  def save_fn(epoch):
    saver.save(sess, path, global_step=epoch)
  return save_fn


def main():
  sess = tf.Session()
  train_fn = train_network(sess)
  validate_fn = validate_network(sess)
  zero_cms_fn = zero_out_confusion_matrix(sess)
  save_fn = save_network(sess)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  for e in range(FLAGS.epoch):
    train_iou, average_loss = train_fn(e)
    validate_iou = validate_fn()
    log_info('[%s] epoch %d average_loss [%f] '
             'train_iou [%f] validate_iou [%f]'
             % (timestamp(), e, average_loss,
                train_iou, validate_iou))
    save_fn(e)
    zero_cms_fn()

if __name__ == '__main__':
    main()
