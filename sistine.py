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
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # release everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
