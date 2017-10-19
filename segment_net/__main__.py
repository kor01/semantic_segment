import tensorflow as tf
from segment_net.train import train_main


def main(_):
  train_main()

if __name__ == '__main__':
    tf.app.run()
