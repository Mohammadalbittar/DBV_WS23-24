import math

import cv2 as cv
import numpy as np
import time
import project.functions_n as fn
#import project.functions_j as fj
import project.Plotten as plot

### TEST ###
#Initialisierung
path = 'resources/video2.mp4'  # Videopfad
ot = fn.Objecttracking()    # ot als Objekt der Klasse Objecttracking definiert
object_detector = cv.createBackgroundSubtractorMOG2(history=60, varThreshold=70)
change_roi = True

#KERNALS
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((3,3),np.uint8)
kernalCl = np.ones((8,8),np.uint8)
fgbg=cv.createBackgroundSubtractorMOG2(detectShadows=True)
kernal_e = np.ones((4,4),np.uint8)

#########

if change_roi:  # Wenn True, kann die roi mit der Funktion ot.set_roi angepasst werden
    ot.set_roi(ot.Imgage_from_Video(path, 100))

#mog = fj.background_sub(methode='mog')
#knn = fj.background_sub(methode='knn')
#cnt = fj.background_sub(methode='cnt')
#gmg = fj.background_sub(methode='gmg')

cap = cv.VideoCapture(path)
start_time = time.time()    # Startzeit des Videos

while True:
    ret, frame = cap.read()
    roi = frame[ot.roi[1]:ot.roi[3], ot.roi[0]: ot.roi[2]] # y1, y2 : x1, x2
    cv.imshow('Region of Interest', roi)

    # Masking methode 1
    mask = object_detector.apply(roi)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    #mask = gray_frame * mask

    # Masking methode 2
    fgmask = fgbg.apply(roi)
    ret, imBin = cv.threshold(fgmask, 254, 255, cv.THRESH_BINARY)
    mask1 = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernalOp)
    mask2 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, kernalCl)
    e_img = cv.erode(mask2, kernal_e)

    #cv.imshow('frame1', frame2)
    cv.imshow('maske', e_img)

    #contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   #mask1
    contours, _ = cv.findContours(e_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #mask2
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

        #area_crossing = [(460, 370), (520, 520), (1150, 460), (940, 345)]
        #test = ot.point_inside_polygon(area_crossing)
        print("")
    print(ot.car_in_out)

    #data_plot.plotData(ot.car_in_out, cap)


    cv.line(frame, (ot.crossing_lines[0][0], ot.crossing_lines[0][1]), (ot.crossing_lines[0][2], ot.crossing_lines[0][3]), (0, 0, 255), 2)  # links
    cv.line(frame, (ot.crossing_lines[1][0], ot.crossing_lines[1][1]), (ot.crossing_lines[1][2], ot.crossing_lines[1][3]), (0, 0, 255), 2)  # unten
    cv.line(frame, (ot.crossing_lines[2][0], ot.crossing_lines[2][1]), (ot.crossing_lines[2][2], ot.crossing_lines[2][3]), (0, 0, 255), 2)  # rechts
    cv.line(frame, (ot.crossing_lines[3][0], ot.crossing_lines[3][1]), (ot.crossing_lines[3][2], ot.crossing_lines[3][3]), (0, 0, 255), 2)  # oben

    cv.imshow('Mask', mask)
    cv.imshow('Mask 2', e_img)
    cv.imshow('Frame', frame)

    key = cv.waitKey(30)
    if key == 27:
        break

end_time = time.time()  # Endzeit des Videos
elapsed_time = end_time - start_time    # Dauer, die das Video abgespielt wurde
cap.release()
cv.destroyAllWindows()
plot.anzahlFahrzeugeProMinute(elapsed_time, len(ot.car_in_out), 0)
plot.anzahlFahrzeugeProRichtung(ot.car_in_out)