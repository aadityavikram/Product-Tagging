import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from keras.preprocessing import image
import glob
import os
import PIL
from PIL import Image
import numpy as np

def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-90, 90)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

def rescaling(image_array: ndarray):
    return sk.transform.rescale(image_array, 1.0/4.0, anti_aliasing=False)

train_path = "dataset\\train"

train_labels = os.listdir(train_path)
image_size = (224, 224)

# dictionary of the transformations functions we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'rescaling' : rescaling
}

for i, label in enumerate(train_labels):
	cur_path = train_path + "\\" + label
	count = 1
	for image_path in glob.glob(cur_path + "\\*.jpg"):
		x = sk.io.imread(image_path)
		transformed_image_0 = available_transformations['rotate'](x)
		transformed_image_1 = available_transformations['noise'](x)
		transformed_image_2 = available_transformations['horizontal_flip'](x)
		transformed_image_3 = available_transformations['rescaling'](x)
		
		new_file_path_0 = '%s/rotated_image_%s.png' % (cur_path, count)
		new_file_path_1 = '%s/noise_filled_image_%s.png' % (cur_path, count)
		new_file_path_2 = '%s/flipped_image_%s.png' % (cur_path, count)
		new_file_path_3 = '%s/rescaled_image_%s.png' % (cur_path, count)

		sk.io.imsave(new_file_path_0, transformed_image_0)
		sk.io.imsave(new_file_path_1, transformed_image_1)
		sk.io.imsave(new_file_path_2, transformed_image_2)
		sk.io.imsave(new_file_path_3, transformed_image_3)
		
		print('Augmented image %s'%count)
		count += 1