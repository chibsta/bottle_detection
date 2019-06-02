# python test_network.py

# import the necessary package

import keras
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import vgg16
from keras.preprocessing.image import load_img, image
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
from sklearn.metrics import classification_report
import h5py
import argparse
import imutils
import cv2
import pickle
import os
import json


# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
# train_path 		= config["train_path"]
test_path 		= config["test_path"]
# features_path 	= config["features_path"]
# labels_path 	= config["labels_path"]
# test_size 		= config["test_size"]
# results 		= config["results"]
model_path 		= config["model_path"]
seed 			= config["seed"]
classifier_path = config["classifier_path"]

# load the trained logistic regression classifier
print ("[INFO] loading the classifier...")
classifier = pickle.load(open(classifier_path, 'rb'))

# pretrained models needed to perform feature extraction on test data too!
if model_name == "vgg16":
	# base_model = VGG16(weights=weights)
	# model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	print("[INFO] loading network...")
	model = VGG16(weights="imagenet", include_top=False)
	image_size = (224, 224)
""" elif model_name == "vgg19":
	base_model = VGG19(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "resnet50":
	base_model = ResNet50(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
	image_size = (224, 224)
elif model_name == "inceptionv3":
	base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (299, 299)
elif model_name == "inceptionresnetv2":
	base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (299, 299)
elif model_name == "mobilenet":
	base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
	model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
	image_size = (224, 224)
elif model_name == "xception":
	base_model = Xception(weights=weights)
	model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
	image_size = (299, 299) 
else:
	base_model = None """

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(model_path, "r")
i = int(db["labels"].shape[0] * 0.75)

# get all the train labels
train_labels = db["label_names"] #[:i]

# get all the test images paths
test_images = os.listdir(test_path)

# loop through each image in the test data
for image_path in test_images:
	path 		= test_path + "/" + image_path
	img 		= image.load_img(path, target_size=image_size)
	x 			= image.img_to_array(img)
	x 			= np.expand_dims(x, axis=0)
	x 			= preprocess_input(x)
	feature 	= model.predict(x)
	flat 		= feature.flatten()
	flat 		= np.expand_dims(flat, axis=0)
	preds 		= classifier.predict(flat)
	prediction 	= train_labels[preds[0]]
	ynew 		= classifier.predict_proba(flat)
	print(ynew)
	# perform prediction on test image
	print ("I think it is a " + prediction) #train_labels[preds[0]])
	img_color = cv2.imread(path, 1)
	cv2.putText(img_color, "I think it is a " + prediction, (140,445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("test", img_color)

	# print(classification_report(db["labels"][i:], preds, target_names=train_labels))

	# key tracker
	key = cv2.waitKey(0) & 0xFF
	if (key == ord('q')):
		cv2.destroyAllWindows()