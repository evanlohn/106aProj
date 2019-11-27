import numpy as np
import cv2
import argparse
import os

cap = cv2.VideoCapture(0)

def mkFileNamer(pth, fname):
    i = 0
    def newFileName():
        nonlocal i
        full_path = os.path.join(pth, fname + str(i) + ".png")
        while os.path.exists(full_path):
            i += 1
            full_path = os.path.join(pth, fname + str(i) + ".png")
        return full_path
    return newFileName

def charTypedWas(typed, c):
    return typed & 0xFF == ord(c)

def main():
    try:
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            color = frame # could also use gray or hsv here
            # Display the resulting frame
            cv2.imshow('frame', color)
            tmp = cv2.waitKey(1)
            #print(ord('q'), tmp, tmp & 0xFF)
            if ret and charTypedWas(tmp, 'c'):
                fname = newFileName()
                print('saving {}'.format(fname))
                cv2.imwrite(fname, color)

            if charTypedWas(tmp,'q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='specify location to save files')
    parser.add_argument('--save', type=str, default='./raw_img_data')
    parser.add_argument('--fname', type=str, default='img')

    args = parser.parse_args()
    if not os.path.isdir(args.save):
        os.mkdir(args.save)
    newFileName = mkFileNamer(args.save, args.fname)
    main()
