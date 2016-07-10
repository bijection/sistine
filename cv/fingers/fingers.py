import cv2, numpy as np, pdb
cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold = 16, detectShadows=True)
sum_ = 0.
n = 0.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
lower = np.array([0, 40, 70], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")
prev = None

def doCalibrateVertical(y, frame):
    pass

VERT_STAGE_SETUP_TIME = 2
VERT_STAGE_TIME = 4
NUM_VERT_STAGES = 12

def verticalCalibration(ind):
    rows,cols,_ = (720, 1280, 3) # frame.shape
    col = cols/2
    rowsToDo = [rows/2 + (i - 5) * 60 for i in range(NUM_VERT_STAGES)]
    vertices = []
    for i in rowsToDo:
        vertices.append(((col, i),(col+50,i+10), i+5)) # top bot vert value

    tl, br, y = vertices[ind]

    def _verticalCalibration(ticks, frame):
        if ticks > VERT_STAGE_SETUP_TIME:
            cv2.rectangle(frame, tl, br,(0,0,255))
            doCalibrateVertical(y, frame)
        else:
            cv2.rectangle(frame, tl, br,(0,255,0))
        
        if ticks > VERT_STAGE_TIME:
            return False
        return True

    return _verticalCalibration

def threshold(ticks, frame):
    return True

# STAGES
STAGES = [verticalCalibration(i) for i in range(NUM_VERT_STAGES)] + [threshold]

n = 0.
initialTicks = cv2.getTickCount()
currStage = STAGES.pop(0)
while (True):
    # init code
    n += 1
    e1 = cv2.getTickCount()

    # stage code
    ret, frame = cap.read()
    ticks = (cv2.getTickCount() - initialTicks)/cv2.getTickFrequency()
    shouldStay = currStage(ticks, frame)
    if not shouldStay:
        currStage = STAGES.pop(0)
        initialTicks = cv2.getTickCount()

    # diagnostics/showing img
    cv2.imshow("images",frame)
    e2 = cv2.getTickCount()
    sum_ += (e2 - e1)/cv2.getTickFrequency()
    print sum_/n

cap.release()
cv2.destroyAllWindows()
