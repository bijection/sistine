import cv2
import numpy as np

# lower = np.array([0, 0, 0], dtype = "uint8")
# upper = np.array([45, 59, 50], dtype = "uint8")

def main():
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture('cv/fingers/fingers.mov')

    # detector = cv2.SimpleBlobDetector()
    # main loop
    while True:
        # frame by frame capture
        # I think there's a callback-based way to do this as well, but I think
        # this way works fine for us
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

        frame = cv2.inRange(frame[:,:,2], 90, 110)

        # frame = cv2.rectangle(frame, (x-100,y-100), (x+100,y+100), 255)
        _, cnts, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        byarea = []
        for c in cnts:
            area = cv2.contourArea(c)
            byarea.append((area, c))
        byarea.sort(key=lambda i: i[0])
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if len(byarea) > 2:
            for i in byarea[-2:]:
                c = i[1]
                hull = cv2.convexHull(c)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    # release everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
