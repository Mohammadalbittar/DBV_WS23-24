import cv2 as cv
import pafy
import pytube
from pytube import YouTube

import sys
import numpy as np
import cv2 as cv
from typing import Union
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hdbscan


def get_video(url:str):
    video = pafy.new(url)       #Create pafy object with youtube URL
    best_qual_stream =video.getbest()   # get the best quality of the stream
    cap = cv.VideoCapture(best_qual_stream.url)        # OpenCV Video Capture Objekt

    return cap      # Return named CV Video Capture Object

def get_livestream(url:str):        #Funktioniert nicht
    youtube_stream = YouTube(url)
    stream = youtube_stream.streams.filter(only_video=True).first()

    cap = cv.VideoCapture(stream.url)
    return cap

# def lukas_kanade(first_image):
#     prvs = cv.cvtColor(first_image, cv.COLOR_BGR2GRAY)
#     hsv = np.zeros_like(prvs)
#     hsv[..., 1] = 255
#     def lukas_kanade2(frame):     # Anpassen an neue Inputs
#         nonlocal prvs
#         nonlocal hsv
#
#         next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.7, 4, 15, 3, 8, 1.2, 0)
#         mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#         hsv[..., 0] = ang*180/np.pi/2
#         hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#         bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#         bgr = cv.addWeighted(bgr, 0.8, frame, 0.5, gamma=0)
#
#         prvs = next
#         return bgr
#
#     return lukas_kanade2


def dense_optical_flow_outer(initial_frame):
    # Convert the initial frame to grayscale
    old_frame = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)

    def dense_optical_flow_inner(frame):
        nonlocal old_frame

        # Convert the current frame to grayscale
        if len(frame.shape) > 2:
            current_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
        else:
            current_frame = frame
            hsv = np.concatenate([frame[:,:,np.newaxis]]*3, axis=-1)

        print(f'hsv shape {hsv.shape}\n')

        # Calculate dense optical flow using Farneback method
        flow = cv.calcOpticalFlowFarneback(old_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert the flow field to polar coordinates
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize the magnitude for better visualization

        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Convert the HSV image to BGR for visualization
        bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        #bgr_flow = cv.addWeighted(bgr_flow, 0.5, frame, 0.8, 0)

        # Update the previous frame
        old_frame = current_frame.copy()

        return frame, bgr_flow

    return dense_optical_flow_inner



def background_sub(methode: Union['mog', 'knn', 'cnt', 'gmg' ]):
    if methode == 'mog':
        backSub = cv.createBackgroundSubtractorMOG2(history=0, varThreshold=50, detectShadows=False)
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
        input_frame = cv.GaussianBlur(frame, (21, 21), 5)
        fg_mask = backSub.apply(input_frame)

        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        #cv.putText(frame, "Frame: {}".format(frame_num), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        result = gray_frame * fg_mask
        #result = cv.bitwise_and(gray_frame, gray_frame, mask=fg_mask)

        return result, fg_mask

    return background_sub2


def motion_extraction(first_frame):
    first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    first_frame = first_frame
    def motion_extraction2(frame):
        nonlocal first_frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.GaussianBlur(frame, (11, 11), 15)
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

def video_tiling_mixed(frame1, frame2, frame3, frame4, width, height):
    frame1 = cv.cvtColor(frame1, cv.COLOR_GRAY2BGR) if len(frame1.shape) == 2 else frame1
    frame2 = cv.cvtColor(frame2, cv.COLOR_GRAY2BGR) if len(frame2.shape) == 2 else frame2
    frame3 = cv.cvtColor(frame3, cv.COLOR_GRAY2BGR) if len(frame3.shape) == 2 else frame3
    frame4 = cv.cvtColor(frame4, cv.COLOR_GRAY2BGR) if len(frame4.shape) == 2 else frame4


    output_frame = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

    output_frame[:height, :width, :3] = frame1
    output_frame[:height, width:, :3] = frame2
    output_frame[height:, :width, :3] = frame3
    output_frame[height:, width:, :3] = frame4

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

def hdbscan_clustering(image, min_cluster_size = 5, min_samples = 3):
    print(f'Type: {type(image)} Shape: {image.shape} Dtype: {image.dtype}')

    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print('Image RGB to Gray in Funktion = hdbscan_clustering')


    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples =min_samples)

    image = image.reshape(-1, 1)
    result = hdbscan_clusterer.fit_predict(image)
    print(f'After FIT Type: {type(result)} Shape: {result.shape} Dtype: {result.dtype}')
    result = result.reshape(image.shape)
    print(f'After Reshape Type: {type(result)} Shape: {result.shape} Dtype: {result.dtype}')
    result = (result * 255 / np.max(result)).astype(np.uint8)

    print(f'Type: {type(result)} Shape: {result.shape} Dtype: {result.dtype}')
    return result



def watershed_segmentation(image):
    if len(image.shape) > 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print(f'Dtype is {gray.dtype}')
        print('Image RGB to Gray in Funktion = wastershed_segmentation')
    else:
        gray = image.astype(np.uint8)

    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=10)

    sure_bg = cv.dilate(opening, kernel, iterations=3)

    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    print(f'markers type {markers.dtype}, marker shape {markers.shape}, image type {image.dtype}, image shape {image.shape}')
    markers = cv.watershed(image, markers)
    markers = markers.astype(np.uint8)
    markers = (markers + 1) * (255 // 2)
    return markers
