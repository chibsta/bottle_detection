# Usage python getframes_process.py

import numpy as np
import cv2
import imutils
import argparse
import zmq
import time
from datetime import datetime
from usbcamerastream import WebcamVideoStream

# Capture the video streams
cam1 = WebcamVideoStream(src=1).start()
cam2 = WebcamVideoStream(src=2).start()
cam3 = WebcamVideoStream(src=3).start()


def send_array_and_str(socket, img1, string, bottle_count, flags=0):
    md1 = dict(dtype=str(img1.dtype), shape=img1.shape)

    socket.send_string(string, flags | zmq.SNDMORE)
    socket.send_json(md1, flags | zmq.SNDMORE)
    socket.send_string(bottle_count, flags | zmq.SNDMORE)
    print("Sent: " + string)
    return socket.send(img1, flags)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:2222")
time.sleep(0.2)

def receive_bp_signal(socketz, flags=0):
    signal = socketz.recv_string(flags=flags)
    return signal

context2 = zmq.Context()
socket2 = context2.socket(zmq.SUB)
socket2.setsockopt(zmq.SUBSCRIBE, b"")
socket2.connect("tcp://192.168.12.175:1111")

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--record", type=str, default="no",
	help="Record a video of the camera(s)")
args = vars(ap.parse_args())

count = 1
bottle_count = 1
old_val = 0
while (True):
    #time.sleep(0.02)
    # Capture frame-by-frame
    #ret, frame1 = cam1.read()
    #ret, frame2 = cam2.read()
    #ret, frame3 = cam3.read()
    
    # send the frames to the detector, if the bottle presence sensor signal has
    # been received
    msg = receive_bp_signal(socket2)
    print(msg)
    #if old_val == 0: # ready to process a new bottle
    if (msg == "1"):
        # # Capture frame-by-frame
        # ret, frame1 = cam1.read()
        frame2 = cam2.read()
        # ret, frame3 = cam3.read()
        print("Bottle present!!")
    #if (count%15 == 0): # Code for testing communication with detection module
        # send_array_and_str(socket, frame1, str(datetime.now()),str(bottle_count))
        send_array_and_str(socket, frame2, str(datetime.now()), str(bottle_count))
        # send_array_and_str(socket, frame3, str(datetime.now()), str(bottle_count))
        cv2.imwrite('/media/storage/cvprojects/bottle_detection/results/foreign_object/foreign_object'+str(datetime.now())+'.jpg', frame2) #foreign_object clean
        #send_array_and_str(socket, frame3, str(datetime.now()))
        #old_val = 1  # set flag to show that we are currently processing a bottle
        bottle_count +=1
        
    elif (msg == "0"):
            print("No bottle present")
            #old_val = 0  # as soon as the bp sensor is off, get ready to take another picture
            
    # Show and write the center frame to video
    #frame2 = imutils.resize(frame2, width=900)
    # write the time on the image

    #cv2.imshow("test", frame1)
    #cv2.imwrite('/media/storage/cvprojects/bottle_detection/results/image'+str(datetime.now())+'.jpg', frame1)
    count += 1
        
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        print("This is where we would have saved something...")
# When everything done, release the capture
cam1.release()
cv2.destroyAllWindows()