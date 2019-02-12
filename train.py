from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import pickle

test_size = 0.30
seed = 9
features_path = "output\\features.h5"
labels_path = "output\\labels.h5"
classifier_path = "output\\classifier.pickle"
train_path = "dataset\\train"

h5f_data  = h5py.File(features_path, 'r')				# import features and labels
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

print("Training started")
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),np.array(labels),test_size=test_size,random_state=seed) # split the training and testing data

model = LogisticRegression(random_state=seed)
print ("Model created")
model.fit(trainData, trainLabels)

preds = model.predict(testData)							#evaluate the model on test data

print("Saving model")
pickle.dump(model, open(classifier_path, 'wb'))			#save the classifier