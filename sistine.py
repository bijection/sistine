import cv2
import numpy as np
import sys

# lower = np.array([0, 0, 0], dtype = "uint8")
# upper = np.array([45, 59, 50], dtype = "uint8")

def apparentlyNotThatSlowThinPointDetector(contour, x, y, w, h):
    buf = np.zeros((h, w))
    cv2.drawContours(buf, [contour], -1, 255, 1, offset=(-x, -y))
    thiny, width = None, float('inf')
    topstart = int(round(h * 0.1))
    bottomstop = int(round(h * 0.9))
    for row in range(topstart, bottomstop):
        for x in range(w):
            if buf[row][x] == 255:
                left = x
                break
        for x in range(w-1, -1, -1):
            if buf[row][x] == 255:
                right = x
                break
        diff = right - left
        if diff < width:
            width = diff
            thiny = row
    return thiny + y # include offset in here

def main():
    cv2.ocl.setUseOpenCL(False)

    if len(sys.argv) >= 2 and sys.argv[1] == 'test':
        cap = cv2.VideoCapture('cv/fingers/fingers.mov')
    else:
        cap = cv2.VideoCapture(0)

    # detector = cv2.SimpleBlobDetector()
    # main loop
    while True:
        # frame by frame capture
        # I think there's a callback-based way to do this as well, but I think
        # this way works fine for us
        ret, frame = cap.read()
        oldframe = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        gray = frame[:,:,0]

        frame = cv2.inRange(frame[:,:,2], 90, 110)

        # frame = cv2.rectangle(frame, (x-100,y-100), (x+100,y+100), 255)
        _, cnts, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        byarea = []
        for c in cnts:
            area = cv2.contourArea(c)
            byarea.append((area, c))
        byarea.sort(key=lambda i: i[0])
        frame = gray
        _, frame = cv2.threshold(frame, 64, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if len(byarea) > 2:
            for i in byarea[-2:]:
                c = i[1]
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                row = apparentlyNotThatSlowThinPointDetector(c, x, y, w, h)
                cv2.line(frame, (x, row), (x + w, row), (255, 0, 0), 2)

        cv2.imshow('frame', frame[:,::-1,:])

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # release everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
