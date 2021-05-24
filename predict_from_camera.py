import tensorflow as tf
from Model.Model import SegmentationModel
import Scripts.DataPrepering as dp
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib import image
from matplotlib import pyplot
from PIL import Image
import cv2
from matplotlib import cm



def main():    
    model = tf.keras.models.load_model("model_with_augumentation_100e_92_77")
    
    vidcap = cv2.VideoCapture(3)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT,30)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH,90)
    success,img = vidcap.read()
    
    img = Image.fromarray(np.uint8(img))
    img = img.resize((128,128))
    img = np.asarray(img).astype('float32')
    print(img.shape)
    img = np.reshape(img, (1,128,128,3))
    
    predict = model.predict(img)
    
    return dp.display([img[0], dp.create_mask(predict)])
    
if __name__ == '__main__':
    main()