import math

import cv2 as cv
import numpy as np
import project.functions_n as fn
import project.data_plot as data_plot

### TEST ###
#Initialisierung
path = r'C:\Noah\Studium Lokal\Master\DBV_Abschlussprojekt\TestVideo1.mp4'  # Videopfad
ot = fn.Objecttracking()    # ot als Objekt der Klasse Objecttracking definiert
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=60)
change_roi = False

#KERNALS
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
fgbg=cv.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((5,5),np.uint8)

#########

if change_roi:  # Wenn True, kann die roi mit der Funktion ot.set_roi angepasst werden
    ot.set_roi(ot.Imgage_from_Video(path, 100))

cap = cv.VideoCapture(path)
while True:
    ret, frame = cap.read()
    roi = frame[ot.roi[1]:ot.roi[3], ot.roi[0]: ot.roi[2]] # y1, y2 : x1, x2
    cv.imshow('Region of Interest', roi)

    # Masking methode 1
    mask = object_detector.apply(roi)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)

    # Masking methode 2
    fgmask = fgbg.apply(roi)
    ret, imBin = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
    mask1 = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernalOp)
    mask2 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, kernalCl)
    e_img = cv.erode(mask2, kernal_e)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   #mask1
    #contours, _ = cv.findContours(e_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #mask2
    bounding_boxes = []

    for cnt in contours:
        area = cv.contourArea(cnt)  # Berechnet die Fläche der erkannten Kontur
        if area > 600:  # Zeichnet ein Rechteck (BB) um die Kontur, wenn die Fläche einen Schwellwert überschreitet
            x, y, w, h = cv.boundingRect(cnt)
            bounding_boxes.append([x, y, w, h]) # Koordinaten der BB in einer Liste speichern

    boxes_ids = ot.add_new_vehicle(bounding_boxes)  # Trackingfunktion auf die BB anwenden
    for box_id in boxes_ids:    # Aktualisiert die gezeichneten BB, Mittelpunkte und Fahrzeug-IDs
        x, y, w, h, cx, cy, id = box_id
        cv.circle(roi, (cx,cy), 5, (0, 0, 255), -1)
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #cv.putText(roi, str(id), (500, 600), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        ot.dist_to_line(0)  # links
        ot.dist_to_line(1)  # unten
        ot.dist_to_line(2)  # rechts
        ot.dist_to_line(3)  # oben
        print("")
    print(ot.car_in_out)

    #data_plot.plotData(ot.car_in_out, cap)


    cv.line(frame, (ot.crossing_lines[0][0], ot.crossing_lines[0][1]), (ot.crossing_lines[0][2], ot.crossing_lines[0][3]), (0, 0, 255), 2)  # links
    cv.line(frame, (ot.crossing_lines[1][0], ot.crossing_lines[1][1]), (ot.crossing_lines[1][2], ot.crossing_lines[1][3]), (0, 0, 255), 2)  # unten
    cv.line(frame, (ot.crossing_lines[2][0], ot.crossing_lines[2][1]), (ot.crossing_lines[2][2], ot.crossing_lines[2][3]), (0, 0, 255), 2)  # rechts
    cv.line(frame, (ot.crossing_lines[3][0], ot.crossing_lines[3][1]), (ot.crossing_lines[3][2], ot.crossing_lines[3][3]), (0, 0, 255), 2)  # oben

    cv.imshow('Mask', mask)
    #cv.imshow('Mask 2', e_img)
    cv.imshow('Frame', frame)

    key = cv.waitKey(30)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
