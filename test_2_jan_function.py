import cv2
import numpy as np


def motion_extraction(frame1, frame2):              # eine Funktion von Jan
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    result = cv2.addWeighted(frame1,    0.5, cv2.bitwise_not(frame2), 0.5, 0)
    result = result.astype(np.float32) / 255
    result = abs(result - 0.5)
    _, result = cv2.threshold(result, 0.35, 1, cv2.THRESH_BINARY)
    result = cv2.dilate(result, None, iterations=10)
    return result

def process_video(video_path):
    cap1 = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)

    cap1.set(cv2.CAP_PROP_POS_FRAMES, 1700)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 1705)

    one_time = True
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames:", total_frames)

    while True:

        ret1, main_capture = cap1.read()
        ret2, sec_capture = cap2.read()
        if not ret1 or not ret2:
            print("Error: Empty frame(s) encountered.")
            break

        if one_time:
            cumulative_result = np.zeros_like(main_capture[:, :, 0], dtype=np.float32)
            _, frame_one = cap1.read()
            one_time = False
        

        # frame_diff = cv2.absdiff(next_capture, main_capture)
        
        # motion = motion_extraction(cv2.GaussianBlur(frame_one, (5, 5), sigmaX=4, sigmaY=5))

        # _, binary_image = cv2.threshold(motion(main_capture), 1, 255, cv2.THRESH_BINARY)
        binary_image = motion_extraction(main_capture, sec_capture)
            
        cumulative_result += binary_image
        cumulative_result = cv2.dilate(cumulative_result, None, iterations=15)
        cumulative_result = cv2.erode(cumulative_result, None, iterations=15)

        # cv2.imshow('Input Video', main_capture)
        # cv2.imshow('Cumulative Difference', cumulative_result)
        # if cv2.waitKey(1) == ord('q'):
        #     break

    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()
    return cumulative_result.astype(np.uint8)


# process_video(r'resources\test_video_3.mp4')
_, process_and_threshold = cv2.threshold(process_video(r'resources\test_video.mp4'), 1, 255, cv2.THRESH_BINARY)


edges = cv2.Canny(process_and_threshold, 100, 120)
edges = cv2.dilate(edges, None, iterations=10)
edges = cv2.erode(edges, None, iterations=7)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 400, 800, 1)
print(lines, 'number: ', len(lines))
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(edges, (x1, y1), (x2, y2), (180, 34, 255), 15)

cv2.imshow('edges', edges)
cv2.waitKey(0)

    