import cv2
import numpy as np
import sys

# parameters
MIDPOINT_DETECTION_DEAD_ZONE = 0.1
FINGER_COLOR_LOW = 90 # b in Lab space
FINGER_COLOR_HIGH = 110 # b in Lab space
MIN_FINGER_SIZE = 7000 # pixels
REFLECTION_MIN_RATIO = 0.1

# unimportant parameters
LINE_WIDTH = 2
CIRCLE_RADIUS = 6
BLUE = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def findTouchPoint(contour, x, y, w, h):
    buf = np.zeros((h, w))
    cv2.drawContours(buf, [contour], -1, 255, 1, offset=(-x, -y))
    thiny, thinx, width = None, None, float('inf')
    topstart = int(round(h * MIDPOINT_DETECTION_DEAD_ZONE))
    bottomstop = int(round(h * (1 - MIDPOINT_DETECTION_DEAD_ZONE)))
    for row in range(topstart, bottomstop + 1):
        left = 0
        for i in range(w):
            if buf[row][i] == 255:
                left = i
                break
        right = w-1
        for i in range(w-1, -1, -1):
            if buf[row][i] == 255:
                right = i
                break
        diff = right - left
        if diff < width:
            width = diff
            thiny = row
            thinx = int(left + diff / 2.0)
    cv2.circle(buf, (thinx, thiny), CIRCLE_RADIUS, BLUE, -1)
    if thiny == topstart or thiny == bottomstop:
        return None, None
    return thiny + y, thinx + x

def main():
    cv2.ocl.setUseOpenCL(False)

    # settings
    if 'test' in sys.argv:
        cap = cv2.VideoCapture('cv/fingers/fingers.mov')
    else:
        cap = cv2.VideoCapture(0)
    orig = 'orig' in sys.argv
    nobox = 'nobox' in sys.argv
    nocontour = 'nocontour' in sys.argv

    # detector = cv2.SimpleBlobDetector()
    # main loop
    while True:
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        # frame by frame capture
        # I think there's a callback-based way to do this as well, but I think
        # this way works fine for us
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.flip(frame, 1) # unmirror left to right
        original = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

        frame = cv2.inRange(frame[:,:,2], FINGER_COLOR_LOW, FINGER_COLOR_HIGH)

        _, cnts, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if orig:
            drawframe = original
        else:
            drawframe  = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        byarea = []
        for c in cnts:
            area = cv2.contourArea(c)
            byarea.append((area, c))
        byarea.sort(key=lambda i: i[0])
        if len(byarea) > 2:
            # is there a finger?
            largest_contour = byarea[-1][1]
            x1, y1, w1, h1 = cv2.boundingRect(largest_contour)
            largest_area = byarea[-1][0]
            if largest_area > MIN_FINGER_SIZE:
                # draw large finger
                if not nocontour:
                    cv2.drawContours(drawframe, [largest_contour], -1, GREEN, LINE_WIDTH)
                if not nobox:
                    cv2.rectangle(drawframe, (x1, y1), (x1 + w1, y1 + h1), RED, LINE_WIDTH)
                # see if there's a reflection
                smaller_contour = byarea[-2][1]
                x2, y2, w2, h2 = cv2.boundingRect(smaller_contour)
                smaller_area = byarea[-2][0]
                # if they overlap in X and the smaller one is above the larger one
                if (not (x1 + w1 < x2 or x2 + w2 < x1)) and y2 + h2 < y1 and \
                        smaller_area / largest_area >= REFLECTION_MIN_RATIO:
                    # hover
                    if not nocontour:
                        cv2.drawContours(drawframe, [smaller_contour], -1, GREEN, LINE_WIDTH)
                    if not nobox:
                        cv2.rectangle(drawframe, (x2, y2), (x2 + w2, y2 + h2), RED, LINE_WIDTH)
                    # TODO better way of estimating this
                    hover_y = ((y1) + (y2 + h2)) / 2.0 # diff between top and bottom
                    hover_x = ((x1 + w1 / 2.0) + (x2 + w2 / 2.0)) / 2.0 # diff between centers
                    hover_x, hover_y = int(hover_x), int(hover_y)
                    cv2.circle(drawframe, (hover_x, hover_y), CIRCLE_RADIUS, BLUE, -1)
                else:
                    # touch
                    # find the touch point height
                    touch_y, touch_x = findTouchPoint(largest_contour, x1, y1, w1, h1)
                    if touch_y is not None:
                        cv2.circle(drawframe, (touch_x, touch_y), CIRCLE_RADIUS, BLUE, -1)

        cv2.imshow('drawframe', drawframe)

    # release everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
