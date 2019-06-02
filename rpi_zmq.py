import zmq
import time  
import curses  
import sys  
import RPi.GPIO as GPIO  

# Setup the GPIO pins
GPIO.setmode(GPIO.BCM)  
GPIO.setup(5, GPIO.IN)  
GPIO.setup(22, GPIO.OUT)

tc=curses.initscr()  
tc.nodelay(1)  
old_bp_sensor_status=GPIO.input(5)  
count=0  

tc.addstr(1, 0, "Press SPACE to quit:\n")  
tc.addstr(2, 0, str(count))

def send_bp_signal(socket, string):
    send_socket.send_string(string)

# ZeroMQ Context for sending bottle presence message
# to take_picture module
send_context = zmq.Context()
send_socket = send_context.socket(zmq.PUB)
send_socket.connect("tcp://*:1111")

# ZeroMQ Context for receiving messages from detection module
receive_context = zmq.Context()
receive_socket = receive_context.socket(zmq.SUB)
# Define subscription and messages with prefix to accept.
receive_socket.setsockopt_string(zmq.SUBSCRIBE, "1")
receive_socket.connect("tcp://10.2.16.27:5680")

while True:
    bp_sensor_status=GPIO.input(5)  
    kbval=tc.getch()    
    if old_bp_sensor_status != bp_sensor_status:  
        time.sleep(0.02) # debounce period  
        bp_sensor_status=GPIO.input(5) # re-read the input  
        if old_bp_sensor_status != bp_sensor_status:  
            if bp_sensor_status == True:  
                tc.addstr(2, 10, "-") # pin is high, no bottle present  
            else:  
                tc.addstr(2, 10, "_")  # pin is low, bottle present
                count = count + 1
                # send signal to take picture
                send_bp_signal(send_socket,"1") 
            old_bp_sensor_status=bp_sensor_status  
            tc.addstr(2, 0, str(count))  
  
    if kbval==0x20:  
        break  
    # Get the message from bottle_detection module
    detection_message= receive_socket.recv()
    if detection_message == "1":
        GPIO.output(22, True)
        print(detection_message)

time.sleep(1)  
GPIO.cleanup()  
curses.endwin()  
print("Goodbye") 
