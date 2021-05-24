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

def main():
    parser = argparse.ArgumentParser(description="Semantic segmentation people from image")
    requiredArgs = parser.add_argument_group("Required argument")
    requiredArgs.add_argument("--input", dest="input", type=str, help="Path to image for segmentation", required=True)
    
    args = parser.parse_args()
    
    model = tf.keras.models.load_model("model_with_augumentation_100e_92_77")
    
    img = Image.open(args.input) 
    img = img.resize((128,128))
    img = np.asarray(img).astype('float32')
    print(img.shape)
    img = np.reshape(img, (1,128,128,3))
    
    predict = model.predict(img)
    
    return dp.display([img[0], dp.create_mask(predict)])
    
if __name__ == '__main__':
    main()