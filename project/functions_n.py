import cv2 as cv
import numpy as np
import math
from scipy.spatial import distance
import time
import os
import matplotlib as plt

class Objecttracking:
    def __init__(self):
        self.center_points = {}
        self.roi = [302, 278, 1278, 567] # x1, y1, x2, y2 - Standardwerte für roi
        self.crossing_lines = [[400, 375, 480, 525], [562, 535, 1215, 475], [980, 345, 1215, 465], [455, 360, 925, 335]] # x1, y1, x2, y2 - links, unten, rechts, oben

        self.id_count = 0
        self.cross_count = 0
        self.car_in_out = np.array([], dtype={'names': ['id', 'in', 'out'], 'formats': ['int', 'int', 'int']})

    def Imgage_from_Video(self, video_path, frame_index):
        cap = cv.VideoCapture(video_path)
        cv.namedWindow('Video', cv.WINDOW_NORMAL)

        if not cap.isOpened():
            print("Fehler beim Öffnen des Videos. Überprüfe den Pfad")
            exit()

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)

        # Frame lesen
        ret, frame = cap.read()

        # Überprüfen, ob das Lesen des Frames erfolgreich war
        if not ret:
            print("Fehler: Frame konnte nicht gelesen werden.")
            return None

        cap.release()
        return frame
    def mouse_callback(self, event, x, y, flags, clicked_points):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"Klick bei x = {x}, y = {y}")
            clicked_points.append(x)
            clicked_points.append(y)
        return 0
    def set_roi(self, img):
        clicked_points = []

        while len(clicked_points) < 3:
            cv.namedWindow("Klickerkennung", cv.WINDOW_AUTOSIZE)
            #cv.resizeWindow("Klickerkennung", 1200, 800)
            print("Obere linke, dann untere rechte Ecke des Bildauschnitts anklicken. Anschliessend mit ESC bestaetigen.")
            cv.setMouseCallback("Klickerkennung", self.mouse_callback, clicked_points)

            while len(clicked_points) < 3:
                cv.imshow("Klickerkennung", img)
                key = cv.waitKey(1) & 0xFF
                if key == 27:  # ESC-Taste
                    break  # Beende die innere Schleife, wenn ESC gedrückt wurde

            if len(clicked_points) == 2:
                break  # Beende die äußere Schleife, wenn genug Punkte geklickt wurden
            cv.destroyAllWindows()  # Schließe das Fenster
            self.roi = clicked_points
            print(self.roi)
        return clicked_points
    def add_new_vehicle(self, obj_rect):

        objects_bbs_ids = []

        # Mittelpunkt berechnen
        for rect in obj_rect:
            x, y, w, h = rect
            cx = int((x + x + w) // 2)
            cy = int((y + y + h) // 2)

            obj_already_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, cx, cy, id])
                    obj_already_detected = True

            # NEW OBJECT DETECTION
            if obj_already_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, cx, cy, self.id_count])
                self.id_count += 1

        # ASSIGN NEW ID to OBJECT
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, object_id = obj_bb_id
            if not np.isin(object_id, self.car_in_out['id']):
                self.car_in_out = np.append(self.car_in_out, np.array((object_id, 0, 0), dtype=self.car_in_out.dtype))
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    def dist_to_line(self, line):
        x1 = self.crossing_lines[line][0]
        y1 = self.crossing_lines[line][1]
        x2 = self.crossing_lines[line][2]
        y2 = self.crossing_lines[line][3]

        for id in self.center_points:
            x0, y0 = self.center_points[id]
            x0 += self.roi[0]
            y0 += self.roi[1]

            dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if np.isin(0, self.car_in_out['in'][id]) and dist < 5:
                index = np.where(self.car_in_out['id'] == id)[0]
                if len(index) > 0:
                    self.car_in_out['in'][index] = line + 1

            if np.isin(0, self.car_in_out['out'][id]) and dist < 5:
                index = np.where(self.car_in_out['id'] == id)[0]
                if len(index) > 0:
                    if np.isin(line+1, self.car_in_out['in'][id]) or np.isin(0, self.car_in_out['in'][id]):
                        None
                    else:
                        self.car_in_out['out'][index] = line + 1

    def calc_dist_obj_line(self, center_points = []):

        if self.center_points is None:
            return None
        # Extrahiere die Koordinaten des Punktes und der Linie
        cx, cy = center_points
        line = self.crossing_lines

        # Berechne die Gleichung der Linie (y = mx + b)
        m = (line[3] - line[1]) / (line[2] - line[0])  # Steigung der Linie
        b = int(line[1] - m * line[0])  # y-Achsenabschnitt der Linie

        m2 = -1 / m # Steigung Grade senkrecht von Punkt auf Linie
        b2 = int(cy / (m2 * cx)) # y-Achsenabschnitt dieser Gerade

        spx = int((b - b2)/(m2 - m))  # x-Koordinate des Schnittpunkts der Geraden
        spy = int(m * spx + b) # y-Koordinate des Schnittpunkts der Geraden

        #distance = math.sqrt((spy - cy)**2 + (spx - cx)**2)
        distance = int(math.dist((spy, spx), (cy, cx)))
        #distance = math.hypot((spx - cx), (spy - cy))
        return distance, cy,cx, spy, spx







'''
    def Get_Images_from_Video(self, video_path, Num_of_Images: int, Time_between_Images: int):
        frame_count = 0
        imgList = []

        cap = cv.VideoCapture(video_path)
        cv.namedWindow('Video', cv.WINDOW_NORMAL)

        if not cap.isOpened():
            print("Fehler beim Öffnen des Videos. Überprüfe den Pfad")
            exit()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            timestamp_ms = cap.get(cv.CAP_PROP_POS_MSEC)

            while frame_count < Num_of_Images:
                if timestamp_ms >= frame_count * Time_between_Images:
                    # Einzelbild speichern
                    # output_path = os.path.join(bilder, f'frame_{frame_count}.jpg')
                    # cv.imwrite(output_path, frame)
                    # print(f'Frame {frame_count} gespeichert.')
                    imgList.append(frame)
                    print(f'Frame {frame_count} zur Liste hinzugefügt')
                    frame_count += 1
                break

            cv.imshow('Video', frame)
            if cv.waitKey(25) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()
        return imgList
'''