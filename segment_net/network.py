from segment_net.flags import *
from segment_net.model import get_network
from collections import namedtuple


OutputTensors = namedtuple(
  'OutputTensors', ('images', 'masks', 'logits', 'softmax',
                    'loss', 'train_op', 'mean_iou', 'update_iou'))


def stable_log(tensor):
  ret = tf.log(tensor + 1e-5)
  return ret


def image_shape():
  return FLAGS.width, FLAGS.height


def segment_loss(sm, classes):
  xent = -stable_log(sm) * classes
  # xent = -tf.log(sm) * classes
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
      shape = (None,) + image_shape() + (3,)
      images = tf.placeholder(
        dtype='float32', shape=shape)
    if masks is None and (iou or train):
      shape = (None,) + image_shape() + (FLAGS.classes,)
      masks = tf.placeholder(
        dtype='float32', shape=shape)

    logits = network(images, FLAGS.classes, is_train=train)

    if iou:
      mean_iou, update_iou = measure_iou(logits, masks)
    else:
      mean_iou, update_iou = None, None

    sm = tf.nn.softmax(logits, dim=-1)

    if train:
      loss = segment_loss(sm, tf.to_float(masks))
      train_op = tf.train.AdamOptimizer(
        learning_rate=FLAGS.lr).minimize(loss)
    else:
      loss, train_op = None, None

    ret = OutputTensors(
      images=images, masks=masks, logits=logits,
      softmax=sm, loss=loss, train_op=train_op,
      mean_iou=mean_iou, update_iou=update_iou)
    return ret


Layer = namedtuple('Layer', ('output_shape',))


class KerasWrapper(object):

  def __init__(self, name, path=None,
               sess=None, reuse=False):
    if sess is None:
      self.graph = tf.Graph()
      self.sess = tf.Session(graph=self.graph)

    else:
      self.sess, self.graph = sess, sess.graph

    with self.graph.as_default():
      self._tensors = build_network(
        name, reuse=reuse, train=False, iou=True)
      pred = tf.argmax(self._tensors.softmax, axis=-1)
      pred = tf.one_hot(pred, 3, axis=-1)

      self._pred_op = pred
      if not reuse:
        self.sess.run(tf.global_variables_initializer())
      self._saver = tf.train.Saver(
        var_list=tf.trainable_variables())
      if path is not None:
        self.load(path)

  def load(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self._saver.restore(
      self.sess, ckpt.model_checkpoint_path)

  @property
  def variables(self):
    with self.graph.as_default():
      variables = tf.trainable_variables()
      values = self.sess.run(variables)
      return list(zip(variables, values))

  @property
  def layers(self):
     return [Layer(output_shape=[FLAGS.width, FLAGS.height, 3])]

  def predict_on_batch(self, x):
    ret = self.sess.run(
      self._pred_op,
      feed_dict={self._tensors.images: x})
    return ret
