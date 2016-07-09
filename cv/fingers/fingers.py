import cv2, numpy as np, pdb
cv2.ocl.setUseOpenCL(False)

cap = cv2.VideoCapture('fingers.mov')
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold = 16, detectShadows=True)
sum_ = 0.
n = 0.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
lower = np.array([0, 40, 70], dtype = "uint8")
upper = np.array([30, 255, 255], dtype = "uint8")

while (True):
    n += 1.
    e1 = cv2.getTickCount()
    ret, frame = cap.read()

    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    fgmask = fgbg.apply(skin)

    cv2.imshow("images",fgmask)
    # frame = cv2.cvtColor(frame_pre, cv2.COLOR_RGB2LAB)[:,:,2]
    
    # cv2.imshow('frame',frame)
    e2 = cv2.getTickCount()
    sum_ += (e2 - e1)/cv2.getTickFrequency()
    print sum_/n
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
