from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import os
import pickle
from PIL import Image

train_path = "dataset\\train"
test_path = "dataset\\test"
classifier_path = "output\\classifier.pickle"

print("Loaded the classifier")														#loaded the trained logistic regression classifier
classifier = pickle.load(open(classifier_path, 'rb'))

base_model = VGG16(weights="imagenet")												#loaded the pretrained vgg16 model
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)	#retraining the last fully connected layer
image_size = (224, 224)

train_labels = os.listdir(train_path)

test_images = os.listdir(test_path)

f=open("predicted_labels.txt",'w+')

for image_path in test_images:
	path = test_path + "\\" + image_path
	try:
		Image.open(path).verify()													#checking if image is corrupt or not
		img = image.load_img(path, target_size=image_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feature = model.predict(x)
		flat = feature.flatten()
		flat = np.expand_dims(flat, axis=0)
		preds = classifier.predict(flat)
		print(image_path + " -> " + train_labels[preds[0]])							#Predicting label of test images
		f.write(image_path + " -> " + train_labels[preds[0]]+'\n')
	except Exception:
		print(image_path + " -> Corrupt Image")
		f.write(image_path + " -> Corrupt Image"+'\n')
f.close()