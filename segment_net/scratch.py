import pickle
import tensorflow as tf


tf.flags.DEFINE_integer('filters', 12, 'number pf filters')
tf.flags.DEFINE_integer('batch_size', 8, 'number pf filters')
tf.flags.DEFINE_string(
  'train', './train_dataset.bin', 'training dataset')
tf.flags.DEFINE_string(
  'validate', './validation_dataset.bin', 'validate dataset')

FLAGS = tf.flags.FLAGS


def stable_log(tensor):
  ret = tf.log(tensor + 1e-5)
  return ret


def encoder_block(img):

  with tf.variable_scope('encoder'):

    conv1 = tf.layers.conv2d(
      img, filters=FLAGS.filters, strides=1, kernel_size=5,
      padding='SAME', activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='SAME')

    conv2 = tf.layers.separable_conv2d(
      pool1, filters=FLAGS.filters, strides=1,
      kernel_size=3, padding='SAME', activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='SAME')

    transpose1 = tf.layers.conv2d_transpose(
      pool1, 32, 2, 2, activation=tf.nn.relu)

    transpose2 = tf.layers.conv2d_transpose(
      pool2, 32, 4, 4, activation=tf.nn.relu)

  return [transpose1, transpose2]


def decode_block(feature_maps):
  with tf.variable_scope('decoder'):
    features = tf.concat(feature_maps, axis=-1)
    features = tf.layers.conv2d(
      features, kernel_size=3, filters=FLAGS.filters,
      padding='SAME', activation=tf.nn.relu)
    logits = tf.layers.conv2d(
      features, filters=3, kernel_size=1, strides=1)
  return logits


def segment_loss(logits, classes):
  sm = tf.nn.softmax(logits, dim=-1)
  xent = -stable_log(sm) * classes
  loss = tf.reduce_sum(xent, axis=3)
  loss = tf.reduce_mean(loss, axis=[0, 1, 2])
  return loss


def build_network(reuse=None):
  with tf.variable_scope('segment_net', reuse=reuse):

    images = tf.placeholder(
      dtype='float32', shape=(None, 256, 256, 3))
    mask = tf.placeholder(
      dtype='int32', shape=(None, 256, 256, 3))

    encodes = encoder_block(images)
    logits = decode_block(encodes)
    loss = segment_loss(logits, tf.to_float(mask))
    train_op = tf.train.AdamOptimizer().minimize(loss)
  return images, mask, logits, loss, train_op


def measure_iou(logits, masks):
  labels = tf.argmax(masks, axis=-1)
  pred = tf.argmax(logits, axis=-1)
  iou = tf.metrics.mean_iou(
    predictions=pred, labels=labels, num_classes=3)
  return iou


def validate_network(sess):
  dataset = pickle.load(open(FLAGS.validate, 'rb'))
  images, masks = dataset['images'], dataset['masks']
  with tf.name_scope('validation'):
    image_ph, mask_ph, logits, _, _ = build_network(reuse=True)
    mean_iou, update_iou = measure_iou(logits, mask_ph)

  batch_size = 16
  iteration = int(len(images) / batch_size) + 1

  iteration = 50

  def validate_fn():
    for i in range(iteration):
      image_batch = images[i * batch_size: (i + 1) * batch_size]
      mask_batch = masks[i * batch_size: (i + 1) * batch_size]
      sess.run(update_iou, {image_ph: image_batch, mask_ph: mask_batch})
    iou_val = sess.run(mean_iou)
    return iou_val

  return validate_fn


def train():
  train_dataset = pickle.load(open(FLAGS.train, 'rb'))
  images, masks = train_dataset['images'], train_dataset['masks']
  image_ph, mask_ph, logits, loss, train_op = build_network()
  mean_iou, update_iou = measure_iou(logits, mask_ph)
  sess = tf.Session()
  validate = validate_network(sess)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  batch_size = FLAGS.batch_size
  total_loss = 0

  for i in range(100):
    image_batch = images[i * batch_size: (i + 1) * batch_size]
    mask_batch = masks[i * batch_size: (i + 1) * batch_size]
    _, loss_val, _, iou_val = sess.run(
      [train_op, loss, update_iou, mean_iou],
      {image_ph: image_batch, mask_ph: mask_batch})
    total_loss += loss_val
    print(total_loss / (i + 1), iou_val)

  print('validate_iou:', validate())


if __name__ == '__main__':
    train()
