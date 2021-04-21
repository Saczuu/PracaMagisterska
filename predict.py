import tensorflow as tf
import Scripts.DataPrepering as dp
from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import random

model = keras.models.load_model('model')
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
train = dataset['train'].map(dp.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(dp.load_image_test)
test_dataset = test.batch(64)
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      dp.display([image[0], mask[0], create_mask(pred_mask)])
  else:
    dp.display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions(test_dataset, 1)