# python test_classifier_on_video.py

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
import time
from imutils.video import VideoStream

# initialize the video stream and allow the camera
# sensor to warmup
print("[INFO] warming up camera...")
vs =  cv2.VideoCapture(0) # VideoStream(src=0).start()

#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)
 
# initialize the FourCC, video writer, dimensions of the frame, and zeros array
fourcc = cv2.VideoWriter_fourcc(*"H264") # cv2.VideoWriter_fourcc(*"MJPG")
writer = None
(h, w) = (None, None)
zeros = None


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
capture_video   = config["capture_video"]
output_video	= config["output_video"]
fps			 = config["fps"]

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

# loop over frames from the video stream
while True:
# loop through each image in the test data
# for image_path in test_images:
# 	path 		= test_path + "/" + image_path
	# Capture frame-by-frame
	ret, frame = vs.read() # use frame = vs.read() if you use the imutils VideoStream class to instantiate the vidoe capture above
	cv2.imshow("test", frame)
	# img 		= image.load_img(frame, target_size=image_size)
	x 			= cv2.resize(frame, image_size, interpolation=cv2.INTER_LINEAR) # image.img_to_array(img)
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
	# print ("I think it is a " + prediction) #train_labels[preds[0]])
	# img_color = cv2.imread(frame, 1)
	# frame = imutils.resize(frame, width=900)
	# cv2.putText(frame, str(prediction) + ": " + str(ynew), (10,645), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
	

	# grab the frame from the video stream and resize it to have a
	# maximum width of 300 pixels
	#frame = vs.read()
	
 
	# check if the writer is None
	if writer is None:
		# store the image dimensions, initialize the video writer,
		# and construct the zeros array
		(h, w) = frame.shape[:2]
		writer = cv2.VideoWriter(output_video, fourcc, fps,
			(w, h), True)
	# 	zeros = np.zeros((h, w), dtype="uint8")

	# 	# break the image into its RGB components, then construct the
	# # RGB representation of each frame individually
	# (B, G, R) = cv2.split(frame)
	# R = cv2.merge([zeros, zeros, R])
	# G = cv2.merge([zeros, G, zeros])
	# B = cv2.merge([B, zeros, zeros])
 
	# # construct the final output frame, storing the original frame
	# # at the top-left, the red channel in the top-right, the green
	# # channel in the bottom-right, and the blue channel in the
	# # bottom-left
	# output = np.zeros((h * 2, w * 2, 3), dtype="uint8") # (h * 2, w * 2, 3), dtype="uint8"
	# output[0:h, 0:w] = frame # 0:h, 0:w
	# output[0:h, w:w * 2] = R
	# output[h:h * 2, w:w * 2] = G
	# output[h:h * 2, 0:w] = B
 
	# write the output frame to file
	writer.write(frame)

	# key tracker
	k = cv2.waitKey(1)

	if k%256 == 27:
		# ESC pressed
		print("Escape hit, closing...")
		break
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.release() # use vs.stop() if you use the imutils VideoStream class to instantiate the vidoe capture above
writer.release()