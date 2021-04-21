import tensorflow as tf
from Model.Model import SegmentationModel
import Scripts.DataPrepering as dp
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt

print("Import complete")
print("Generating Model")

model = SegmentationModel.generateModel()

print("Compiling model")
print("==============")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


print("Loading and preparing data")
print("==============")

img_height = 334
img_width = 500
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
train = dataset['train'].map(dp.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(dp.load_image_test)
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


print("Training model")
print("==============")
EPOCHS = 25
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset)
                        
model.save("model")