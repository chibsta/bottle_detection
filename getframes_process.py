# Usage python getframes_process.py

import numpy as np
import cv2
import imutils
import argparse
import zmq
import time

# Capture the video streams
cam1 = cv2.VideoCapture(0)

def send_array_and_str(socket, img, string, btn, flags=0):
    md = dict(dtype = str(img.dtype), shape=img.shape)

    socket.send_string(string, flags | zmq.SNDMORE)
    socket.send_string(btn, flags | zmq.SNDMORE)
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(img, flags)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5667")
time.sleep(0.2)

def recv_array_and_str(socketz, flags=0, copy=True, track=False):
    string = socketz.recv_string(flags=flags)
    btn = socketz.recv_string(flags=flags)
    md = socketz.recv_json(flags=flags)
    msg = socketz.recv(flags=flags, copy=copy, track=track)

    img = np.frombuffer(bytes(memoryview(msg)), dtype=md['dtype'])
    return string, btn, img.reshape(md['shape'])

context2 = zmq.Context()
socket2 = context2.socket(zmq.SUB)
socket2.setsockopt(zmq.SUBSCRIBE, b"")
socket2.connect("tcp://localhost:5667")

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--record", type=str, default="no",
	help="Record a video of the camera(s)")
args = vars(ap.parse_args())



while(True):
    # Capture frame-by-frame
    ret, frame1 = cam1.read()
    print(ret)
    frame1 = imutils.resize(frame1, width=900)
    cv2.imshow("test", frame1)
    # if some condition is met send frame via ZMQ
    send_array_and_str(socket, frame1, "12", "Detecton Frame")

    print(recv_array_and_str(socket2)[0])

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