import cv2
import numpy as np

def process_video(video_path):
    cap1 = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)


    first_frame = 100

    cap1.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, first_frame + 50)

    one_time = True

    while True:

        _, main_capture = cap1.read()
        _, next_capture = cap2.read()

        main_capture = cv2.cvtColor(main_capture, cv2.COLOR_BGR2GRAY)
        next_capture = cv2.cvtColor(next_capture, cv2.COLOR_BGR2GRAY)

        if one_time:

            cumulative_result = np.zeros_like(main_capture, dtype=np.float32)
            
            one_time = False

        frame_diff = cv2.absdiff(next_capture, main_capture)

        _, binary_image = cv2.threshold(frame_diff, 1, 255, cv2.THRESH_BINARY)

        cumulative_result += binary_image

        cv2.imshow('Input Video', main_capture)
        cv2.imshow('Cumulative Difference', cumulative_result.astype(np.uint8))
        if cv2.waitKey(1) == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

process_video(r'resources\test_video.mp4')