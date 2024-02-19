import sys

import numpy as np
import cv2 as cv
from typing import Union


def lukas_kanade(path):     # Anpassen an neue Inputs
    cap = cv.VideoCapture(cv.samples.findFile(path))
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while(1):
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.7, 4, 15, 3, 8, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        bgr = cv.addWeighted(bgr, 0.8, frame2, 0.5, gamma=0)
        cv.imshow('frame2', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', frame2)
            cv.imwrite('opticalhsv.png', bgr)
        prvs = next
    cv.destroyAllWindows()



def background_sub(methode: Union['mog', 'knn', 'cnt', 'gmg' ]):
    if methode == 'mog':
        backSub = cv.createBackgroundSubtractorMOG2(history=0, varThreshold=50, detectShadows=True)
        print('MOG 2 Selected')
    elif methode == 'knn':
        backSub = cv.createBackgroundSubtractorKNN()
        print('KNN Selected')
    elif methode == 'cnt':
        backSub = cv.bgsegm.createBackgroundSubtractorCNT()
        print('CNT Selected')
    elif methode == 'gmg':
        backSub = cv.bgsegm.createBackgroundSubtractorGMG()
        print('GMG Selected')
    else:
        print('No viable method selected')
        sys.exit()
    def background_sub2(frame):
        input_frame = cv.GaussianBlur(frame, (3, 3), 3)
        fg_mask = backSub.apply(input_frame)

        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        #cv.putText(frame, "Frame: {}".format(frame_num), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #result = gray_frame * fg_mask
        result = cv.bitwise_and(gray_frame, gray_frame, mask=fg_mask)

        return result, fg_mask

    return background_sub2


def motion_extraction(first_frame):
    first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    first_frame = first_frame
    def motion_extraction2(frame):
        nonlocal first_frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if first_frame is not None:
            result = cv.addWeighted(first_frame, 0.5, cv.bitwise_not(frame), 0.5, 0)
            #result = abs(result - (255//2))
            first_frame = frame
            return result
        else: print('No motion detected')
    return motion_extraction2




def stitch_frames(frame1, frame2, frame3, frame4):
    # Assuming all frames have the same height
    height = frame1.shape[0]

    # Concatenate frames horizontally
    stitched_frame = np.concatenate((frame1, frame2, frame3, frame4), axis=1)

    return stitched_frame

def video_tiling(frame1, frame2, frame3, frame4, width, height, Color:str = True):

    if Color == True:
        output_frame = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    else:
        output_frame = np.zeros((height * 2, width * 2), dtype=np.uint8)

    output_frame[:height, :width] = frame1
    output_frame[:height, width:] = frame2
    output_frame[height:, :width] = frame3
    output_frame[height:, width:] = frame4

    return output_frame


def add_text_to_frame(frame, text):
    position = (100, 100)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)
    thickness = 2

    result_frame = frame.copy()
    cv.putText(result_frame, text, position, font, font_scale, color, thickness, cv.LINE_AA)
    return result_frame