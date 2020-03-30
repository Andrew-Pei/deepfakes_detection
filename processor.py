"""
Process an image that we can pass to our networks.
"""
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
from PIL import Image

count=0
def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    global count
    # Load the image.
    img=cv2.imread(image)

    h, w, _ = target_shape
    image1 = load_img(image, target_size=(h, w))
    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image1)
    #str1="/mnt/pic/"+image[31:]

    #if count < 10:
    #    count+=1
    #    cv2.imwrite('image_'+ str(count) + '.jpg', img_arr)
    x = (img_arr / 255.).astype(np.float32)

    return x
