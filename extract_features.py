from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import h5py
import os
import json

train_path = "dataset\\train"
if os.path.exists("output")==False:
	os.system("mkdir " + "output")
features_path = "output\\features.h5"
labels_path = "output\\labels.h5"
test_size = 0.30
model_path = "output\\model"
base_model = VGG16(weights="imagenet")												#loading pretrained vgg16 model
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)	#retraining the last fully connected layer
image_size = (224, 224)

print ("Loaded the model")

train_labels = os.listdir(train_path)

le = LabelEncoder()																	#encode the labels
le.fit([z for z in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
count = 1
for i, label in enumerate(train_labels):
	cur_path = train_path + "\\" + label
	count = 1
	for e in [cur_path + "\\*.jpg",cur_path + "\\*.png"]:
		for image_path in glob.glob(e):
			img = image.load_img(image_path, target_size=image_size)
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			feature = model.predict(x)
			flat = feature.flatten()
			features.append(flat)
			labels.append(label)
			print ("Processed image %s in this category"%count)
			count += 1

le = LabelEncoder()																	#encode the labels
le_labels = le.fit_transform(labels)

h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))						#saving the features

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))						#saving the labels

h5f_data.close()
h5f_label.close()

model_json = model.to_json()														#saving model
with open(model_path + str(test_size) + ".json", "w") as json_file:
	json_file.write(model_json)
print("Saved model")

model.save_weights(model_path + str(test_size) + ".h5")								#saving weights
print("Saved weights")