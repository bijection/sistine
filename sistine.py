import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    # main loop
    while True:
        # frame by frame capture
        # I think there's a callback-based way to do this as well, but I think
        # this way works fine for us
        ret, frame = cap.read()

        # display
        # cv2.imshow('frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        # frame[:,:,0] = 0
        # frame[:,:,1] = 0
        # frame[:,:,2] = 255.0 * (frame[:,:,2] > 100)

        x,y = cv2.minMaxLoc(frame[:,:,2])[2]
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2RGB)
        frame = cv2.rectangle(frame, (x-100,y-100), (x+100,y+100), 255)


        # l, a, b = cv2.split(frame)
        # l = cv2.threshold(l, 100)
        # frame = cv2.cvtColor(frame, cv2.COLOR_LAB2RGB)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # release everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
