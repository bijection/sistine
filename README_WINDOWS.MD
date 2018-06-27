# Project Windows-Touch

![Sistine * 3/2](splash.png)

Turn a Windows-PC into a touchscreen using only $1.0 of hardware. The proof-of-concept, dubbed “Project Sistine” after our [recreation](https://www.anishathalye.com/media/2018/04/03/thumbnail.jpg) of the famous [painting](https://en.wikipedia.org/wiki/The_Creation_of_Adam) in the Sistine Chapel, was prototyped by [Anish Athalye](https://www.anishathalye.com/), [Kevin Kwok](https://twitter.com/antimatter15), [Guillermo Webster](https://twitter.com/biject), and [Logan Engstrom](https://github.com/lengstrom) in about 16 hours. It was thus mordified by [Ashish Gupta](https://github.com/ashishgupta1350) to work from mac/linux to windows.

## Basic Principle

The basic principle behind Sistine is simple. Surfaces viewed from an angle tend to look shiny, and you can tell if a finger is touching the surface by checking if it’s touching its own reflection.

![Hover versus touch](https://www.anishathalye.com/media/2018/04/03/explanation.png)

Kevin, back in middle school, noticed this phenomenon and built [ShinyTouch](https://antimatter15.com/project/shinytouch/), utilizing an external webcam to build a touch input system requiring virtually no setup. We wanted to see if we could miniaturize the idea and make it work without an external webcam. Our idea was to retrofit a small mirror in front of a MacBook’s built-in webcam, so that the webcam would be looking down at the computer screen at a sharp angle. The camera would be able to see fingers hovering over or touching the screen, and we’d be able to translate the video feed into touch events using computer vision.

(Read the rest of our blog post, including a video demo and a high-level explanation of the algorithm, (https://www.anishathalye.com/2018/04/03/macbook-touchscreen/)

## Installation (with pip Python)

Download Python 3 and then pip.

* Install Python 3 via webisite https://www.python.org/download/releases/3.0/ 

* Download and Install pip. Follow this video : https://www.youtube.com/watch?v=AVCcFyYynQY

* Install OpenCV 3 via pip with `pip install opencv-python`

* Install pyautogui via pip with `pip install pyautogui`

* Install Numpy via pip with `pip install numpy`


## Running

Run `python WindowsTouch.py` via cmd or compile using python3. Don't forget to add python 3 to path. 
Python 2 is supported.
