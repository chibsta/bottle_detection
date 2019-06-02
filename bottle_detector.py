import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import vgg16
from keras.preprocessing.image import load_img, image
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
import h5py

import numpy as np
import cv2
import imutils
import zmq
import time

import pickle
# import os
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


# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(model_path, "r")
i = int(db["labels"].shape[0] * 0.75)

# get all the train labels
train_labels = db["label_names"]  #[:i]

# Broadcast rejection message
def send_rejection_message(message):
    socket.send_string(message)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:3333")
time.sleep(0.2)

def recv_array_and_str(socketz, flags=0, copy=True, track=False):
    string = socketz.recv_string(flags=flags)
    
    # Receive first image
    md1 = socketz.recv_json(flags=flags)
    count = socketz.recv_string(flags=flags)
    msg1 = socketz.recv(flags=flags, copy=copy, track=track)
    img1 = np.frombuffer(bytes(memoryview(msg1)), dtype=md1['dtype'])
    
    print("Received: " + string)
    return string, img1.reshape(md1['shape']), str(count)

context2 = zmq.Context()
socket2 = context2.socket(zmq.SUB)
socket2.setsockopt(zmq.SUBSCRIBE, b"")
socket2.connect("tcp://localhost:2222")

current_counter = 0

while (True):
    #for im in range (1,3):
    arr         = recv_array_and_str(socket2)
    frame       = arr[1] 
    x 			= cv2.resize(frame, image_size, interpolation=cv2.INTER_LINEAR) # image.img_to_array(img)
    x 			= np.expand_dims(x, axis=0)
    x 			= preprocess_input(x)
    feature 	= model.predict(x)
    flat 		= feature.flatten()
    flat 		= np.expand_dims(flat, axis=0)
    preds 		= classifier.predict(flat)
    prediction 	= train_labels[preds[0]]
    #ynew 		= classifier.predict_proba(flat)
    print("Bottle: " + arr[2] + ": " + prediction)

    # if prediction is any of the dirty categories send signal to Pi
    send_rejection_message(str(1))
