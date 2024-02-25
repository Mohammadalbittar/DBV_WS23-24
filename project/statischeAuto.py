import cv2
import numpy as np
import functions_n as fn
import math

ot = fn.Objecttracking()
test_file = r'C:\Users\mosta\Videos\2024-02-10 17-21-23.mp4'
cap = cv2.VideoCapture(test_file)
bg_image = r"C:\Users\mosta\PycharmProjects\pythonProject\Result_Background.jpg"
img = cv2.imread(bg_image)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)
frame_counter = 0
point_Stat = []
while True:
    frame_counter +=1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
    ret, frame = cap.read()
    if not ret:
        break
    frame2 = frame
    #Apply Bg Subractor
    #frame = cv2.medianBlur(frame,7)
    #element = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
    #frame = cv2.dilate(frame, element)

    #fgmask = fgbg.apply(frame)
    #fgmask= cv2.medianBlur(fgmask,7)

    #contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #find difference between bg and current frame
    frame_g = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame_g,img)
    diff[diff<70] = 0
    diff[diff>0] = 255

    #diff_still = cv2.absdiff(fgmask,diff)
    #diff_still = cv2.medianBlur(diff_still,5)
    #diff = cv2.GaussianBlur(diff,(3,3),0)
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for cnt in contours:
        # Calculate the center of mass for each contour
        area = cv2.contourArea(cnt)
        if area > 600:
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, w, h])

    if frame_counter > 1:
        old_boxes = ot.center_points
        boxes_ids = ot.add_new_vehicle(bounding_boxes)
        for old in old_boxes.items():
            id_O = old[0]
            cx_O, cy_O = old[1]
            for new in boxes_ids:
                x, y, w, h, cx, cy, id = new
                if id_O == id:
                    print(id)
                    print("====================")
                    dist = math.hypot(cx_O-cx,cy_O-cy)
                    print(dist)
                    if dist == 0:
                        print("didnt move")
                        point_Stat.append([cx_O, cy_O])
                    else:
                        print("did move")
    else:
        boxes_ids = ot.add_new_vehicle(bounding_boxes)

    #for box_id in boxes_ids:
        #x, y, w, h, cx, cy, id = box_id
        #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        #cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for pt in point_Stat:
        cv2.circle(frame2, (pt[0], pt[1]), 5, (0, 255, 0), -1)






    cv2.imshow('Original Frame', frame)
    #cv2.imshow('Foreground Mask', fgmask)
    cv2.imshow("gefiltert",diff)
    cv2.imshow("Not Moving",frame2)



    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()