import numpy as np
import cv2 as cv

test_file = r'C:\Users\mosta\Videos\2024-02-10 17-21-23.mp4'
cap = cv.VideoCapture(test_file)

num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
#Sample frames
frames_ids = []
sampling_interval = 500
frame_id = 0
frames = []

while frame_id < num_frames:
    frames_ids.append(frame_id)
    frame_id += sampling_interval

print(frames_ids)
for i in frames_ids:
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    frames.append(frame)

print(np.array(frames[1]).shape)
bg = np.median(frames,axis=0).astype(np.uint8)
cv.imshow('NO Background',bg)
cv.imwrite("Result_Background.jpg",bg)
cv.waitKey(0)