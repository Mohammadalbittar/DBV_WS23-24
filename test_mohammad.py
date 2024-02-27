import cv2
import numpy as np
from collections import Counter

def majority_frame(frames):
    # Assuming all frames are of the same shape
    h, w = frames[0].shape
    majority_frame = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            pixel_values = [frame[i, j] for frame in frames]
            majority_value = Counter(pixel_values).most_common(1)[0][0]
            majority_frame[i, j] = majority_value

    return majority_frame

# Function to process the video
def process_video(video_path):
    # Open the video
    cap1 = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)


    # Festlegen des ersten Frames
    first_frame = 100

    # Set the starting frame
    cap1.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, first_frame + 1)

    one_time = True
    cumulative_result = []

    for _ in range(1000):

        # Read the first frame
        _, main_capture = cap1.read()
        _, next_capture = cap2.read()

        # Convert the first frame to grayscale
        main_capture = cv2.cvtColor(main_capture, cv2.COLOR_BGR2GRAY)
        next_capture = cv2.cvtColor(next_capture, cv2.COLOR_BGR2GRAY)

        # if one_time:

        #     # Initialize an empty array for the cumulative result
        #     cumulative_result = np.zeros_like(main_capture, dtype=np.float32)
            
        #     one_time = False

        ## Subtract the current frame from the last frame
        # frame_diff = cv2.absdiff(next_capture, main_capture)
        frame_diff = next_capture - main_capture

        # Update the cumulative result
        _, binary_image = cv2.threshold(frame_diff, 1, 255, cv2.THRESH_BINARY)

        # cumulative_result += binary_image

        cumulative_result.append(binary_image)

        # Display the current frame and the cumulative result
        # cv2.imshow('Input Video', main_capture)
        # cv2.imshow('Cumulative Difference', cumulative_result.astype(np.uint8))

        # Update the last frame
        # last_frame = gray_frame

        # print(cumulative_result)
    
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    # print(cumulative_result)
    result = majority_frame(cumulative_result)

    cv2.imshow('cumu', result)
    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)

    # After a key is pressed, close the window and release resources
    cv2.destroyAllWindows()
    

# Replace 'path_to_video.mp4' with the path to your video file
process_video(r'C:\Users\moham\Downloads\DBV_WS23-24\resources\test_video.mp4')
