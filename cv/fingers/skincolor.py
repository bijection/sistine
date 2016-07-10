''' Detect human skin tone and draw a boundary around it.
Useful for gesture recognition and motion tracking.

Inspired by: http://stackoverflow.com/a/14756351/1463143

Date: 08 June 2013
'''

# Required moduls
import cv2
import numpy
cv2.ocl.setUseOpenCL(False)


# Constants for finding range of skin color in YCrCb
min_YCrCb = numpy.array([0,133,77],numpy.uint8)
max_YCrCb = numpy.array([255,173,127],numpy.uint8)

# Create a window to display the camera feed
cv2.namedWindow('Camera Output')

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)
bgst = cv2.createBackgroundSubtractorMOG2(history=500)

# Process the video frames
keyPressed = -1 # -1 indicates no key pressed

while(keyPressed < 0): # any key pressed has a value >= 0

    # Grab video frame, decode it and return next video frame
    ret, frame = videoFrame.read()
    frame = bgst.apply(frame)

    cv2.imshow('Camera Output',frame)

    # Check for user input to close program

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
