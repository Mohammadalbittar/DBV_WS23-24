import cv2 as cv
import pafy
from pytube import YouTube
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import torch

import sys
import numpy as np
from typing import Union
import hdbscan


def get_video(url:str):
    """
    Retrieves a video from a given URL and returns a VideoCapture object.

    Parameters:
    url (str): The URL of the video.

    Returns:
    cv.VideoCapture: The VideoCapture object representing the video.
    """
    video = pafy.new(url)  # Create pafy object with youtube URL
    streams = video.streams
    print(streams)
    if len(streams) >= 1:
        second_lowest_qual_stream = streams[-1]  # Choosing the lowest quality stream
    else:
        second_lowest_qual_stream = video.getbest()  # If only one stream is available, use it

    cap = cv.VideoCapture(second_lowest_qual_stream.url)  # OpenCV Video Capture Object

    return cap  # Return named CV Video Capture Object

def get_livestream(url:str):
    """
    Retrieves the livestream from the given URL.

    Parameters:
    url (str): The URL of the livestream.

    Returns:
    cv.VideoCapture: The captured livestream.
    """
    youtube_stream = YouTube(url)
    stream = youtube_stream.streams.filter(only_video=True).first()

    cap = cv.VideoCapture(stream.url)
    return cap


def dense_optical_flow_outer(initial_frame):
    """
    Calculate dense optical flow between consecutive frames.

    Args:
        initial_frame (numpy.ndarray): The initial frame.

    Returns:
        function: A function that takes a frame as input and returns the frame, flow image, magnitude, and angle mask.
    """
    # Lukas Kanade Dense optical flow. Returns the frame, a flow image, the magnitude and the angle mask
    # Convert the initial frame to grayscale
    old_frame = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)

    def dense_optical_flow_inner(frame):
            """
            Calculate dense optical flow between the previous frame and the current frame.

            Parameters:
            frame (numpy.ndarray): The current frame.

            Returns:
            tuple: A tuple containing the following:
                - frame (numpy.ndarray): The original frame.
                - bgr_flow (numpy.ndarray): The optical flow visualization in BGR format.
                - mag (numpy.ndarray): The magnitude of the optical flow.
                - ang (numpy.ndarray): The angle of the optical flow.
            """
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

            return frame, bgr_flow, mag, ang

    return dense_optical_flow_inner



def background_sub(methode: Union['mog', 'knn', 'cnt', 'gmg']):
    """
    Applies background subtraction to a video frame using the specified method.

    Parameters:
    - methode: The method to be used for background subtraction. It can be one of the following:
        - 'mog': MOG2 method
        - 'knn': KNN method
        - 'cnt': CNT method
        - 'gmg': GMG method

    Returns:
    - background_sub2: A function that takes a video frame as input and applies background subtraction to it.

    Note:
    - The returned function, background_sub2, applies Gaussian blur, extracts the foreground mask, and performs bitwise operations on the frame.

    Example usage:
    ```
    # Create a background subtraction function using MOG2 method
    bg_sub_mog = background_sub('mog')

    # Apply background subtraction to a video frame
    result, fg_mask = bg_sub_mog(frame)
    ```
    """
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
        """
        Apply background subtraction to the input frame.

        Parameters:
        frame (numpy.ndarray): The input frame.

        Returns:
        tuple: A tuple containing the result of background subtraction and the foreground mask.
        """
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
    """
    Extracts motion from consecutive frames using the first frame as a reference.

    Parameters:
    first_frame (numpy.ndarray): The first frame of the video sequence.

    Returns:
    function: A function that takes a frame as input and returns the motion extraction result.

    """
    first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    first_frame = first_frame
    def motion_extraction2(frame):
        """
        Extracts motion from a given frame using background subtraction.

        Parameters:
        frame (numpy.ndarray): The input frame.

        Returns:
        numpy.ndarray: The binary motion mask.
        """
        nonlocal first_frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.GaussianBlur(frame, (11, 11), 15)
        if first_frame is not None:
            result = cv.addWeighted(first_frame,    0.5, cv.bitwise_not(frame), 0.5, 0)
            result = result.astype(np.float32) / 255
            result = abs(result - 0.5)
            _, result = cv.threshold(result, 0.05, 1, cv.THRESH_BINARY)
            result = cv.dilate(result, None, iterations=5)
            first_frame = frame
            return result
        else:
            print('No motion detected')
    return motion_extraction2




def stitch_frames(frame1, frame2, frame3, frame4):
    """
    Stitch together four frames horizontally.

    Parameters:
    frame1 (numpy.ndarray): The first frame.
    frame2 (numpy.ndarray): The second frame.
    frame3 (numpy.ndarray): The third frame.
    frame4 (numpy.ndarray): The fourth frame.

    Returns:
    numpy.ndarray: The stitched frame.
    """
    # Assuming all frames have the same height
    height = frame1.shape[0]

    # Concatenate frames horizontally
    stitched_frame = np.concatenate((frame1, frame2, frame3, frame4), axis=1)

    return stitched_frame

def video_tiling(frame1, frame2, frame3, frame4, width, height, Color: str = True):
    """
    Tiles four input frames into a single output frame.

    Args:
        frame1 (numpy.ndarray): The first input frame.
        frame2 (numpy.ndarray): The second input frame.
        frame3 (numpy.ndarray): The third input frame.
        frame4 (numpy.ndarray): The fourth input frame.
        width (int): The width of each input frame.
        height (int): The height of each input frame.
        Color (bool, optional): Indicates whether the output frame should be in color or grayscale. 
                                Defaults to True (color).

    Returns:
        numpy.ndarray: The tiled output frame.
    """

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
    """
    Combines four input frames into a single output frame using tiling.

    Args:
        frame1 (numpy.ndarray): The first input frame.
        frame2 (numpy.ndarray): The second input frame.
        frame3 (numpy.ndarray): The third input frame.
        frame4 (numpy.ndarray): The fourth input frame.
        width (int): The width of each input frame.
        height (int): The height of each input frame.

    Returns:
        numpy.ndarray: The output frame with the four input frames tiled together.
    """
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

def video_tiling_mixed(frame1, frame2, width, height):
    """
    Combines four input frames into a single output frame using tiling.

    Args:
        frame1 (numpy.ndarray): The first input frame.
        frame2 (numpy.ndarray): The second input frame.
        width (int): The width of each input frame.
        height (int): The height of each input frame.

    Returns:
        numpy.ndarray: The output frame with the two input frames tiled together.
    """
    frame1 = cv.cvtColor(frame1, cv.COLOR_GRAY2BGR) if len(frame1.shape) == 2 else frame1
    frame2 = cv.cvtColor(frame2, cv.COLOR_GRAY2BGR) if len(frame2.shape) == 2 else frame2

    output_frame = np.zeros((height*2, width, 3), dtype=np.uint8)

    output_frame[:height, :width, :3] = frame1
    output_frame[height:, :width, :3] = frame2

    return output_frame

def add_text_to_frame(frame, text):
    """
    Adds text to a given frame at a specified position.

    Args:
        frame (numpy.ndarray): The input frame to add text to.
        text (str): The text to be added to the frame.

    Returns:
        numpy.ndarray: The frame with the text added.
    """
    position = (100, 100)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)
    thickness = 2

    result_frame = frame.copy()
    cv.putText(result_frame, text, position, font, font_scale, color, thickness, cv.LINE_AA)
    return result_frame


def hdbscan_clustering(image, min_cluster_size=5, min_samples=3):
    """
    Perform HDBSCAN clustering on an image.

    Parameters:
    - image: numpy.ndarray
        The input image.
    - min_cluster_size: int, optional
        The minimum number of samples required for a cluster.
    - min_samples: int, optional
        The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    - result: numpy.ndarray
        The input image with clusters overlayed.
    """
    print(f'Type: {type(image)} Shape: {image.shape} Dtype: {image.dtype}')

    if len(image.shape) > 2:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print('Image RGB to Gray in Function = hdbscan_clustering')
    else:
        image_gray = image

    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    image_flat = image_gray.reshape(-1, 1)
    cluster_labels = hdbscan_clusterer.fit_predict(image_flat)
    cluster_labels = cluster_labels.reshape(image_gray.shape)

    # Create a mask where each cluster is assigned a unique color
    cluster_mask = np.zeros_like(image)
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label == -1:  # Noise points
            continue
        cluster_mask[cluster_labels == label] = np.random.randint(0, 255, 3)

    # Overlay the cluster mask on the original image
    result = cv.addWeighted(image, 0.7, cluster_mask, 0.3, 0)

    print(f'Type: {type(result)} Shape: {result.shape} Dtype: {result.dtype}')
    return result



def watershed_segmentation(image):
    """
    Perform watershed segmentation on an input image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The segmented image.

    """
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


def yolo_region_count(region_points=  [(700, 800), (1900, 700), (1600, 500), (600, 550)]):
    """
    Nested function for YOLO and Object Counter, with region points as input parameter.
    
    Args:
        region_points (list): List of tuples representing the region points in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    
    Returns:
        function: A function that takes a frame as input and returns the result of object counting within the specified region.

    Die Umsetzung dieser Funktion ist an verschiedenen Online Beispielen orientiert und notwenige Stellen wurden für unsere Umsetzung abgewandelt.
    """
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using mps')
    else:
        device = torch.device('cpu')
        print("No GPU available. Running on CPU.")
    classes = [2,3,5,7]
    model = YOLO("yolov8n.pt").to(device)
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False, reg_pts=region_points, classes_names=model.names, draw_tracks=True)

    def yolo_region_count2(frame):
        """
        Performs object tracking using the YOLO model and counts the number of objects in a given frame.

        Args:
            frame: The input frame for object tracking and counting.

        Returns:
            result: The count of objects in the frame.

        """
        inside = counter.in_counts
        outside = counter.out_counts
        tracks = model.track(frame, persist=True, show=False,classes = [2,3,5,7])
        result = counter.start_counting(frame, tracks)
        return result, inside, outside
    return yolo_region_count2

def yolo_predict():
    """
    Function that initializes a YOLO model and returns a prediction function.

    Returns:
    yolo_predict2: A function that takes a frame as input and returns the predicted results.
    """

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using mps')
    else:
        device = torch.device('cpu')
        print("No GPU available. Running on CPU.")

    model = YOLO("yolov8n.pt").to(device)
    # {0: 'person',
    #  1: 'bicycle',
    #  2: 'car',
    #  3: 'motorcycle',
    #  4: 'airplane',
    #  5: 'bus',
    #  6: 'train',
    #  7: 'truck',
    #  8: 'boat',
    #  9: 'traffic light', .... }
    classes = [2,3,5,7]
    def yolo_predict2(frame):
        """
        Perform object detection using the YOLO model on a given frame.

        Args:
            frame: The input frame for object detection.

        Returns:
            The results of object detection.
        """
        results = model.predict(frame, stream_buffer=True, classes=classes)
        results = results[0].plot(conf=False, labels=False)
        return results
    def yolo_predict2(frame):
        results = model.predict(frame, stream_buffer = True, classes = classes)
        results = results[0].plot(conf = False, labels = False)
        return results
    return yolo_predict2

def yolo_track():
    """
    Function that performs object tracking using YOLO model.

    Returns:
    yolo_track2 (function): Function that takes a frame as input and returns the frame with tracked objects, 
                            as well as the x and y coordinates of the last tracked object.
    """
    model = YOLO("yolov8n.pt")
    classes = [2, 3, 5, 7]
    def yolo_track2(frame):
        """
        Perform YOLO object detection on a given frame.

        Args:
            frame: The input frame for object detection.

        Returns:
            frame: The modified frame with bounding boxes and circles drawn around detected objects.
            x: The x-coordinate of the last detected object.
            y: The y-coordinate of the last detected object.
        """
        results = model.predict(frame, classes=classes, stream_buffer=True)
        boxes = results[0].boxes.xyxy
        frame = results[0].plot(conf=False, labels=False)
        for box in boxes:
            x, y, w, h = box
            x,y = box_middle(x,y,w,h)
            frame = cv.circle(frame, (x, y), 10, (0, 0, 255), -1)

        return frame
    return yolo_track2



def box_middle(x, y, w, h):
    """
    Calculate the coordinates of the middle point of a box.

    Parameters:
    x (int): The x-coordinate of the top-left corner of the box.
    y (int): The y-coordinate of the top-left corner of the box.
    w (int): The width of the box.
    h (int): The height of the box.

    Returns:
    tuple: The coordinates of the middle point of the box as a tuple (x, y).
    """
    return (int(x - w//2), int(y - h//2))
def box_middle(x,y,w,h):
    w = w-x
    h = h-y
    return (int(x + w//2), int(y + h//2))

def noah_coord_trafo(x,y,w,h):
    """
    Calculate the coordinates of the middle point of a box.

    Parameters:
    x (int): The x-coordinate of the top-left corner of the box.
    y (int): The y-coordinate of the top-left corner of the box.
    w (int): The width of the box.
    h (int): The height of the box.

    Returns:
    tuple: The coordinates of the middle point of the box as a tuple (x, y).
    """
    return (int(x), int(y), int(w-x), int(h-y))