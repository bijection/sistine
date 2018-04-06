# Project Sistine

![Sistine * 3/2](splash.png)

We turned a MacBook into a touchscreen using only $1 of hardware and a little bit of computer vision. The proof-of-concept, dubbed “Project Sistine” after our [recreation](https://www.anishathalye.com/media/2018/04/03/thumbnail.jpg) of the famous [painting](https://en.wikipedia.org/wiki/The_Creation_of_Adam) in the Sistine Chapel, was prototyped by [Anish Athalye](https://twitter.com/anishathalye), [Kevin Kwok](https://twitter.com/antimatter15), [Guillermo Webster](https://twitter.com/biject), and [Logan Engstrom](https://github.com/lengstrom) in about 16 hours.

## Basic Principle

The basic principle behind Sistine is simple. Surfaces viewed from an angle tend to look shiny, and you can tell if a finger is touching the surface by checking if it’s touching its own reflection.

![Hover versus touch](https://www.anishathalye.com/media/2018/04/03/explanation.png)

Kevin, back in middle school, noticed this phenomenon and built [ShinyTouch](https://antimatter15.com/project/shinytouch/), utilizing an external webcam to build a touch input system requiring virtually no setup. We wanted to see if we could miniaturize the idea and make it work without an external webcam. Our idea was to retrofit a small mirror in front of a MacBook’s built-in webcam, so that the webcam would be looking down at the computer screen at a sharp angle. The camera would be able to see fingers hovering over or touching the screen, and we’d be able to translate the video feed into touch events using computer vision.


## Installation

* Install OpenCV 3 using __brew install opencv3__ or whatever

## Running

__/usr/bin/python sistine.py__  # this uses the "system" Python provided by Apple which already includes [Quartz module](https://pypi.org/project/pyobjc-framework-Quartz)

If instead you want to use a brew-installed Python then you may need to:
* __pip2 install pyobjc-framework-Quartz__  # required only once
* __python2 sistine.py__
