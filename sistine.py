import cv2
import numpy as np
import sys, pdb
import pickle
import simulate

# dont change parameters
COMP_DIMENSION_X = 1440
COMP_DIMENSION_Y = 900

# parameters
MIDPOINT_DETECTION_SKIP_ZONE = 0.08
MIDPOINT_DETECTION_IGNORE_ZONE = 0.1
FINGER_COLOR_LOW = 90 # b in Lab space
FINGER_COLOR_HIGH = 110 # b in Lab space
MIN_FINGER_SIZE = 7000 # pixels
REFLECTION_MIN_RATIO = 0.1
FINGER_WIDTH_LOCATION_RATIO = 0.5 # percent of way down from point to dead space
MOVING_AVERAGE_WEIGHT = 0.5

CAPTURE_DIMENSION_X = 1280
CAPTURE_DIMENSION_Y = 720

WINDOW_SHIFT_X = (COMP_DIMENSION_X - CAPTURE_DIMENSION_X)/2
WINDOW_SHIFT_Y = (COMP_DIMENSION_Y - CAPTURE_DIMENSION_Y)/2

CALIBRATION_X_COORDS = [.3,.5,.7]
CALIBRATION_Y_COORDS = [.5,.7,.9]

VERT_STAGE_SETUP_TIME = 3
VERT_STAGE_TIME = 6

# unimportant parameters
LINE_WIDTH = 2
LINE_HEIGHT = 100
CIRCLE_RADIUS = 6
FINGER_RADIUS = 40
PURPLE = (255, 0, 255)
CYAN = (255, 255, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CALIB_CIRCLE_RADIUS = 10

def segmentImage(image):
    # this is kinda wrong cause image is actually BGR
    # but apparently it works??
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    image = cv2.inRange(image[:,:,2], FINGER_COLOR_LOW, FINGER_COLOR_HIGH)
    return image

def opencv2system(ox, oy):
    return (ox + WINDOW_SHIFT_X, oy + WINDOW_SHIFT_Y)

def findTouchPoint(contour, x, y, w, h):
    buf = np.zeros((h, w))
    cv2.drawContours(buf, [contour], -1, 255, 1, offset=(-x, -y))
    thiny, thinx, width = None, None, float('inf')
    topstart = int(round(h * MIDPOINT_DETECTION_SKIP_ZONE))
    bottomstop = int(round(h * (1 - MIDPOINT_DETECTION_SKIP_ZONE)))
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
    cv2.circle(buf, (thinx, thiny), CIRCLE_RADIUS, PURPLE, -1)
    validstart = int(round(h * MIDPOINT_DETECTION_IGNORE_ZONE))
    validstop = int(round(h * (1 - MIDPOINT_DETECTION_IGNORE_ZONE)))
    if not (validstart < thiny < validstop):
        return None, None, None, None

    width_row = int(thiny + FINGER_WIDTH_LOCATION_RATIO * (validstop - thiny))
    left = 0
    for i in range(w):
        if buf[width_row][i] == 255:
            left = i
            break
    right = w-1
    for i in range(w-1, -1, -1):
        if buf[width_row][i] == 255:
            right = i
            break
    widthloc = x + left
    width = right - left
    return thinx + x, thiny + y, widthloc, width


def findHoverPoint(
        contour_big,
        x1,
        y1,
        w1,
        h1,
        contour_small,
        x2,
        y2,
        w2,
        h2):
    # this can probably be done more efficiently...
    buf1 = np.zeros((h1, w1))
    cv2.drawContours(buf1, [contour_big], -1, 255, 1, offset=(-x1, -y1))
    left1 = 0
    for i in range(w1):
        if buf1[0][i] == 255:
            left1 = i
            break
    right1 = w1 - 1
    for i in range(w1-1, -1, -1):
        if buf1[0][i] == 255:
            right1 = i
            break
    mid1 = left1 + (right1 - left1) / 2.0

    buf2 = np.zeros((h2, w2))
    cv2.drawContours(buf2, [contour_big], -2, 255, 2, offset=(-x2, -y2))
    left2 = 0
    for i in range(w2):
        if buf2[-1][i] == 255:
            left2 = i
            break
    right2 = w2 - 1
    for i in range(w2-1, -1, -1):
        if buf2[-1][i] == 255:
            right2 = i
            break
    mid2 = left2 + (right2 - left2) / 2.0

    mid_y = ((y1) + (y2 + h2)) / 2.0
    mid_x = ((x1 + mid1) + (x2 + mid2)) / 2.0
    return int(mid_x), int(mid_y)


# find finger and touch / hover points in an image
# debugframe is the thing to draw on
# returns x, y, touch
# x and y and touch are none if nothing is found
# touch is true if it's a touch, otherwise it's false
def find(segmented_image, debugframe=None, options={}):
    found_x, found_y, touch = None, None, None
    if cv2.__version__.startswith('4'):
        cnts, _ = cv2.findContours(segmented_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, cnts, _ = cv2.findContours(segmented_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            # see if there's a reflection
            smaller_contour = byarea[-2][1]
            x2, y2, w2, h2 = cv2.boundingRect(smaller_contour)
            smaller_area = byarea[-2][0]
            # if they overlap in X and the smaller one is above the larger one
            if (not (x1 + w1 < x2 or x2 + w2 < x1)) and y2 + h2 < y1 and \
                    smaller_area / largest_area >= REFLECTION_MIN_RATIO:
                # hover
                if debugframe is not None:
                    if not options['nocontour'] and not options['nodemodebug']:
                        cv2.drawContours(debugframe, [largest_contour], -1, GREEN, LINE_WIDTH)
                        cv2.drawContours(debugframe, [smaller_contour], -1, GREEN, LINE_WIDTH)
                    if not options['nobox'] and not options['nodemodebug']:
                        cv2.rectangle(debugframe, (x1, y1), (x1 + w1, y1 + h1), RED, LINE_WIDTH)
                        cv2.rectangle(debugframe, (x2, y2), (x2 + w2, y2 + h2), RED, LINE_WIDTH)
                hover_x, hover_y = findHoverPoint(largest_contour, x1, y1, w1, h1,
                        smaller_contour, x2, y2, w2, h2)
                return hover_x, hover_y, False
            else:
                # touch
                # find the touch point height
                touch_x, touch_y, wloc, width = findTouchPoint(largest_contour, x1, y1, w1, h1)
                if touch_y is not None:
                    if debugframe is not None:
                        if not options['nocontour'] and not options['nodemodebug']:
                            cv2.drawContours(debugframe, [largest_contour], -1, GREEN, LINE_WIDTH)
                        if not options['nobox'] and not options['nodemodebug']:
                            cv2.rectangle(debugframe, (x1, y1), (x1 + w1, y1 + h1),
                                    RED, LINE_WIDTH)
                        if not options['nowidth']:
                            cv2.line(debugframe, (wloc, touch_y + LINE_HEIGHT), (wloc, touch_y - LINE_HEIGHT),
                                    BLUE, LINE_WIDTH)
                            cv2.line(debugframe, (wloc + width, touch_y + LINE_HEIGHT),
                                    (wloc + width, touch_y - LINE_HEIGHT), BLUE, LINE_WIDTH)
                    return touch_x, touch_y, True
    return None, None, None

def calibration(ind):
    rows,cols,_ = (720, 1280, 3) # frame.shape
    col = cols/2

    pts = []
    for i in range(len(CALIBRATION_X_COORDS)):
        x_frac = CALIBRATION_X_COORDS[i]
        for j in range(len(CALIBRATION_Y_COORDS)):
            if j == 0 and i != 1:
                continue
            y_frac = CALIBRATION_Y_COORDS[j]
            x = int(x_frac * CAPTURE_DIMENSION_X)
            y = int(y_frac * CAPTURE_DIMENSION_Y)
            pt = (x,y)
            pts.append(pt)

    pt = pts[ind]
    x_calib, y_calib = pt

    def _calibration(segmented, debugframe, options, ticks, drawframe, calib, state):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if ticks > VERT_STAGE_SETUP_TIME:
            cv2.putText(drawframe, 'Keep touching the dot', (10,50), font, 1, (255,255,255),2,cv2.LINE_AA)
            cv2.circle(drawframe, (x_calib, y_calib), CALIB_CIRCLE_RADIUS, RED, -1)
            x, y, touch = find(segmented, debugframe=drawframe, options=options)
            if touch is not None:
                cv2.circle(drawframe, (x,y), CIRCLE_RADIUS, PURPLE, -1)
                calib['calibrationPts'][ind].append((x,y))
        else:
            cv2.putText(drawframe, 'Move your finger to the dot', (10,50), font, 1, (255,255,255),2,cv2.LINE_AA)
            cv2.circle(drawframe, (x_calib, y_calib), CALIB_CIRCLE_RADIUS, GREEN, -1)
        if ticks > VERT_STAGE_TIME:
            # cleanup
            calib['realPts'][ind] = pt
            return False
        return True

    return _calibration

def mainLoop(segmented, debugframe, options, ticks, drawframe, calib, state):
    if 'initialized' not in state:
        nnn = (None, None, None)
        state['last'] = [nnn, nnn, nnn] # last 3 results
        state['last_drawn'] = None # a pair (x, y)
        state['initialized'] = True
        state['md'] = False
        state['usemouse'] = False

    x, y, touch = find(segmented, debugframe=drawframe, options=options)
    state['last'].append((x, y, touch))
    state['last'].pop(0)
    if 'hom' not in calib:
        webcam_points = calib['calibrationPts']
        real_points = calib['realPts']
        calib['orp'] = real_points
        screen_points = []
        for i in range(len(real_points)):
            for _ in range(len(webcam_points[i])):
                screen_points.append(real_points[i])

        webcam_points = [i for s in webcam_points for i in s]
        hom = findTransform(webcam_points, screen_points)
        calib['hom'] = hom
        if not ('nocalib' in sys.argv):
            pickle.dump(calib, open('previous.pickle','w+'))

    if not options['nocalib']:
        for i, j in calib['orp']:
            i_, j_ = applyTransform(i, j, np.linalg.inv(calib['hom']))
            cv2.circle(drawframe, (i, j), CIRCLE_RADIUS, RED, -1)
            cv2.line(drawframe, (i, j), (i_, j_), RED, LINE_WIDTH)
    
    shouldMouse = state['usemouse']
    font = cv2.FONT_HERSHEY_SIMPLEX

    if shouldMouse:
        cv2.putText(drawframe, 'Mouse on', (10,50), font, 1, (255,255,255),2,cv2.LINE_AA)
    else:
        cv2.putText(drawframe, 'Mouse off', (10,50), font, 1, (255,255,255),2,cv2.LINE_AA)

    if touch is not None:
        if not options['demo']:
            cv2.circle(drawframe, (x, y), CIRCLE_RADIUS, PURPLE, -1)
        x_, y_ = applyTransform(x, y, calib['hom'])
        if state['last_drawn'] is not None:
            x_ = int(x_ * MOVING_AVERAGE_WEIGHT + (1 - MOVING_AVERAGE_WEIGHT) * state['last_drawn'][0])
            y_ = int(y_ * MOVING_AVERAGE_WEIGHT + (1 - MOVING_AVERAGE_WEIGHT) * state['last_drawn'][1])
        state['last_drawn'] = (x_, y_)
        cv2.circle(drawframe, (x_, y_), FINGER_RADIUS, CYAN, -1)
        
        mx, my = opencv2system(x_,y_)
        if shouldMouse:
            simulate.mousemove(mx, my)
        if touch:
            if not state['md'] and shouldMouse:
                simulate.mousedown(mx, my)
                state['md'] = True
            cv2.circle(drawframe, (x_, y_), FINGER_RADIUS, YELLOW, -1)
        else:
            if state['md'] and shouldMouse:
                simulate.mouseup(mx, my)
                state['md'] = False
            cv2.circle(drawframe, (x_, y_), FINGER_RADIUS, CYAN, -1)
        cv2.circle(drawframe, (x_, y_), CIRCLE_RADIUS, GREEN, -1)
    else:
        state['last_drawn'] = None

    return True

# points are in the format [(x, y)]
def findTransform(webcam_points, screen_points):
    print webcam_points
    print screen_points
    webcam_points = np.array(webcam_points).astype(np.float)
    screen_points = np.array(screen_points).astype(np.float)
    hom, mask = cv2.findHomography(webcam_points, screen_points, method=cv2.RANSAC)
    return hom


# returns the transformed (x, y) as a pair
def applyTransform(x, y, homography):
    inp = np.array([[[x, y]]], dtype=np.float)
    res = cv2.perspectiveTransform(inp, homography)
    x_, y_ = res[0,0]
    return int(round(x_)), int(round(y_))


CALIBRATION_MESSAGE = """
During the calibration process, you will be 
shown a series of 9 dots on your screen.

Touch the dot with your finger and hold your
finger at that position until the next dot 
appears.

When the dot is green, it is not capturing
your finger position. When the dot is green,
it is recording information.

Enable/disable mouse control with "k" key.

When you are ready to begin, press space bar.
"""

def waitSetup(segmented, debugframe, options, ticks, drawframe, calib, state):
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(drawframe, 'Sistine Calibration', (10,50), font, 2, (255,255,255),2,cv2.LINE_AA)
    for i, line in enumerate(CALIBRATION_MESSAGE.split('\n')):
        cv2.putText(drawframe, line, (10,150 + i * 30), font, 1, (255,255,255),2,cv2.LINE_AA)

    return True


def main():
    cv2.ocl.setUseOpenCL(False) # some stuff dies if you don't do this

    initialStageTicks = cv2.getTickCount()
    calib = {
        "calibrationPts":[[] for i in range(9)],
        "realPts":[(0,0)] * 7
    }

    if 'nocalib' in sys.argv:
        with open('previous.pickle') as f:
            calib = pickle.load(f)
        stages = [mainLoop]
    else:
        stages = [waitSetup] + [calibration(i) for i in range(7)] + [mainLoop]

    currStage = stages.pop(0)

    # settings
    options = {}
    if 'test' in sys.argv:
        cap = cv2.VideoCapture('cv/fingers/fingers.mov')
    else:
        cap = cv2.VideoCapture(0)
    options['orig'] = 'orig' in sys.argv
    options['nobox'] = 'nobox' in sys.argv
    options['nocontour'] = 'nocontour' in sys.argv
    options['nowidth'] = 'nowidth' in sys.argv
    options['nocalib'] = 'nocalib' in sys.argv
    options['demo'] = 'demo' in sys.argv
    options['nodemodebug'] = 'nodemodebug' in sys.argv
    if options['demo']:
        options['nocontour'] = True
        options['nowidth'] = True
        options['nobox'] = True
        options['nocalib'] = True

    debugframe = None
    # main loop
    state = {}
    while True:
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
        elif key & 0xff == ord('k'):
            state['usemouse'] = not state['usemouse']
        elif key & 0xff == ord(' ') and currStage == waitSetup:
            currStage = stages.pop(0)
            initialStageTicks = cv2.getTickCount()

        # frame by frame capture
        # I think there's a callback-based way to do this as well, but I think
        # this way works fine for us
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.flip(frame, 1) # unmirror left to right
        segmented = segmentImage(frame)

        # only matters for debugging
        if options['orig']:
            drawframe = frame
        elif options['demo']:
            drawframe = np.zeros_like(frame)
        else:
            drawframe = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)

        ticks = (cv2.getTickCount() - initialStageTicks)/cv2.getTickFrequency()
        if not currStage(segmented, debugframe, options, ticks, drawframe, calib, state):
            currStage = stages.pop(0)
            initialStageTicks = cv2.getTickCount()
        
        cv2.imshow('drawframe', drawframe)
        cv2.moveWindow('drawframe', WINDOW_SHIFT_X, WINDOW_SHIFT_Y)

    # release everything
    cap.release()
    cv2.destroyAllWindows()
    #if state['md']:
    #    simulate.mouseup(mx, my)

if __name__ == '__main__':
    main()
