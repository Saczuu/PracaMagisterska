import cv2
from time import sleep
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import shutil
from PIL import Image
import tensorflow as tf

"""
    Pakiet 'DataPreparing' odpowiada za dostep do kamery uzytkownika, przechwytywanie obrazu z urzadzenia
    oraz za zapisanie przechwyconych ramek.
    Pakiet ten dodatkowo umozliwia przetworzenie zapisanych ramek na macierze liczbowe.
"""
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask += 1
    return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask
    
def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]
    
def captureFrame(path_for_save,number_of_frame_to_capture = 100, sleep_time = 2, number_of_device = 0):
    """
    Funckja 'captureFrame' służy do pobierania obrazu video z karery urządzenia o numerze 'number_of_device',
    obraz wideo otrzymywany z urządzenia zapisywany jest w formacie .jpg.
    Kolejne ramki pobierane sa co 'sleep_time'.
    Ilość ramek pobieranych okreslamy parametrem 'number_of_frame_to_capture'.
    Plki ze zdjeciami zapisywane sa w folderze definiowanym parametrem 'path_to_save'.
    """ 
        
    # ilosc plikow o nazwie frame* w podanym przez uzytkownika folderze.
    count = 0
    for root, dirs, files in os.walk(path_for_save+"/"):  
        for filename in files:
            if filename[:5] == "frame":
               count += 1
    recived_images = 0
    # Przechwytywanie i zapisywanie obrazu, pliki beda zpisane pod nazwa frame*.jpg
    vidcap = cv2.VideoCapture(number_of_device)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT,30)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH,90)
    success,image = vidcap.read()
    while recived_images < number_of_frame_to_capture:
        success,image = vidcap.read()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path_for_save+"/frame%d.jpg" % count, gray_image)     # save frame as JPEG file
        count += 1
        recived_images += 1
        sleep(2)
            
    vidcap.release() # Zamkniecie kamery
    
def loadImages(path_to_folder):
    """
    Funckcja zwrca zdjecia z zadanego folderu w formie tf.variable.
    Zdjecia zwrocone sa w jednej liscie.
    """
    array = []
    for root, dirs, files in os.walk(path_to_folder+"/"):
        for filename in files:
            file = np.asarray(cv2.imread(path_to_folder+"/"+filename))
            if file.shape != ():
                image = tf.io.read_file(path_to_folder+"/"+filename)
                image = tf.image.decode_jpeg(image, channels=3) 
                array.append(image)
    return array
def loadMasks(path_to_folder):
    """
    Funckcja zwrca maski do segmentacji obrazu z zadanego folderu w formie tf.variables.
    Zdjecia zwrocone sa w jednej liscie.
    """
    array = []
    for root, dirs, files in os.walk(path_to_folder+"/"):
        for filename in files:
            file = np.asarray(cv2.imread(path_to_folder+"/"+filename))
            if file.shape != ():
                mask = tf.io.read_file(path_to_folder+"/"+filename)
                mask = tf.image.decode_jpeg(mask, channels=1) 
                array.append(mask)
    return array


def convertIntoArray(path_to_folder):
    """
    Funckcja zwrca zdjecia z zadanego folderu w formie macierzy liczbowej z 3 kanalami kolorowymi.
    Zdjecia zwrocone sa w jednej macierzy numpy.array
    """
    array = []
    for root, dirs, files in os.walk(path_to_folder+"/"):
        for filename in files:
            file = np.asarray(cv2.imread(path_to_folder+"/"+filename))
            if file.shape != ():
                array.append(file)         
    return np.array(array, dtype='float32')
    
    
def resizeImage(path_to_image, height, width, save = True):
    """
    Funkcja zmienia rozmiar zdjecia w podanej lokalizacji nastepnie nadpisuje plik z grafika.
    """
    image = cv2.imread(path_to_image)
    resized_image = cv2.resize(image, (width, height))
    if save:
        return cv2.imwrite(path_to_image, resized_image)
    else:
        return np.array(resized_image)
    
def resizeImagesInFolder(path_to_folder, width, height):
    for root, dirs, files in os.walk(path_to_folder+"/"):
        for directory in dirs:
            if (directory[0] == "."):
                continue
            for root2, dirs2, files2 in os.walk(path_to_folder+"/"+directory+"/"):
                for filename in files2:
                    if filename[-3:] == "jpg" or filename[-3:] == "png" and "checkpoint" not in filename:
                        resizeImage(path_to_folder+"/"+directory+"/"+filename, width, height)
        for filename in files:
            if filename[-3:] == "jpg" or filename[-3:] == "png" and "checkpoint" not in filename:
                resizeImage(path_to_folder+"/"+filename, width, height)
                                    
def copyFilesToAnotherFolder(path_to_orgianals, path_for_copies, name_format_for_copies):
    iter_for_name = 0
    for root, dirs, files in os.walk(path_to_orgianals+"/"):
        for directory in dirs:
            for root, dirs, files in os.walk(path_to_orgianals+"/"+directory+"/"):
                for filename in files:
                    if filename[-3:] == "jpg" and "checkpoint" not in filename:
                        image = cv2.imread(path_to_orgianals+"/"+directory+"/"+filename)
                        if np.array(image,dtype=np.float64).sum() > 1:
                            iter_for_name += 1
                            cv2.imwrite(path_for_copies+"/"+name_format_for_copies+str(iter_for_name)+".jpg", np.array(image,dtype=np.float64))
        for filename in files:
            if filename[-3:] == "jpg" and "checkpoint" not in filename:
                image = cv2.imread(path_to_orgianals+"/"+filename)
                if np.array(image,dtype=np.float64).sum() > 1:
                    iter_for_name += 1
                    cv2.imwrite(path_for_copies+"/"+name_format_for_copies+str(iter_for_name)+".jpg", np.array(image,dtype=np.float64))