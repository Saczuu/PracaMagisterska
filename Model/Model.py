import tensorflow as tf
from keras.models import *
from keras.layers import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SegmentationModel:

    def generateModel(input_height=360, input_width=480):

        # Przygotowania pliku wsadowego
        input_image = tf.keras.Input(shape=((128, 128, 3)))

        # Przygotowania warstw urkytych modelu
        #Decoder
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
        conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        #Bottom
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        
        #Encoder
        up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

        up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        conv5 = Dropout(0.2)(conv5)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
        
        #Output
        out = Conv2D( 3, (1, 1) , padding='same')(conv5)

        model = tf.keras.Model(input_image ,  out)
        
        model._name = "SaczewskiMaciej"

        return model


