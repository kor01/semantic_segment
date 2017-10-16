from matplotlib import pylab as plt


def show_image_mask(batch, idx, cls=None):
  image, mask = batch[0][idx], batch[1][idx]
  image, mask = image.astype('uint8'), mask.astype('uint8')
  mask[:, :, :] *= 255
  plt.subplot(1, 2, 1)
  plt.imshow(image)
  plt.subplot(1, 2, 2)
  if cls is not None:
    mask = mask[:, :, cls]
  plt.imshow(mask)
  plt.show()
