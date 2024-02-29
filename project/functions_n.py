import cv2 as cv
import numpy as np
import math
from scipy.spatial import distance
import time
import os
import matplotlib as plt

class Objecttracking:   # Die Klasse beinhaltet Funktionen für das Tracken der Fahrzeuge mit openCV, das Berechnen der Abstände zu den Linien und das Zählen der Überquerten Fahzeuge
    def __init__(self):
        self.center_points = {} # Punkte in der Mitte der BoundingBoxes mit der Fahrzeug-ID im Dictionary gespeichert
        self.roi = [302, 278, 1278, 567] # x1, y1, x2, y2 - Standardwerte für roi
        self.crossing_lines = [[400, 375, 480, 525], [562, 535, 1215, 475], [980, 345, 1215, 465], [455, 360, 925, 335]] # x1, y1, x2, y2 - links, unten, rechts, oben

        self.id_count = 0   # Counter für das hinzufügen neuer Fahrzeuge
        self.car_in_out = np.array([], dtype={'names': ['id', 'in', 'out'], 'formats': ['int', 'int', 'int']})  # Array in dem Fahrzeug-ID und die Ein-/ Ausfahrt auf die Kreuzung gespeichert werden

    def Imgage_from_Video(self, video_path, frame_index):   # Mit der FUnktion kann ein Frame zu einer Bestimmten Zeit aus dem Video extrahiert werden
        cap = cv.VideoCapture(video_path)
        cv.namedWindow('Video', cv.WINDOW_NORMAL)

        if not cap.isOpened():
            print("Fehler beim Öffnen des Videos. Überprüfe den Pfad")
            exit()

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)    # Setzt den Frame auf die gewünschte Zeit
        ret, frame = cap.read() # Frame lesen

        # Überprüfen, ob das Lesen des Frames erfolgreich war
        if not ret:
            print("Fehler: Frame konnte nicht gelesen werden.")
            return None

        cap.release()
        return frame    # Gibt den Frame zurück
    def mouse_callback(self, event, x, y, flags, clicked_points):   # Mit der Funktion können die Koordinaten eines Klicks auf einem Bild in einer Liste gespeichert werden
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"Klick bei x = {x}, y = {y}")
            clicked_points.append(x)
            clicked_points.append(y)
        return 0
    def set_roi(self, img): # Mit der Funktion kann die Region of Interest angepasst werden
        clicked_points = []

        # Erstellen eines Fensters mit dem Bild in dem die Funktion mouse_callback funktioniert
        while len(clicked_points) < 3:
            cv.namedWindow("Klickerkennung", cv.WINDOW_AUTOSIZE)
            #cv.resizeWindow("Klickerkennung", 1200, 800)
            print("Obere linke, dann untere rechte Ecke des Bildauschnitts anklicken. Anschliessend mit ESC bestaetigen.")
            cv.setMouseCallback("Klickerkennung", self.mouse_callback, clicked_points)

            # Öfnnen des Fensters
            while len(clicked_points) < 3:
                cv.imshow("Klickerkennung", img)
                key = cv.waitKey(1) & 0xFF
                if key == 27:  # ESC-Taste
                    break  # Beendet die innere Schleife, wenn ESC gedrückt wurde

            if len(clicked_points) == 2:
                break  # Beendet die äußere Schleife, wenn 2 Punkte geklickt wurden
            cv.destroyAllWindows()  # Schließt das Fenster
            self.roi = clicked_points   # Die geklickten Punkte werden an die roi Variable der Klasse übergeben
            print(self.roi)
        return clicked_points   # optionale Rückgabe der Werte
    def add_new_vehicle(self, obj_rect):    # Diese Funktion ist für das Tracken eines Fahrzeugs und neuer Fahrzeuge zuständig

        objects_bbs_ids = []    # Hier werden Daten zu dem gleichen Fahrzeug gespeichert und aktualisiert
        area_crossing = [460, 370, 520, 520, 1150, 460, 940, 345]
        # Berechnen des Mittelpunktes der BoundingBox
        for rect in obj_rect:
            x, y, w, h = rect
            cx = int((x + x + w) // 2)
            cy = int((y + y + h) // 2)

            obj_already_detected = False    # Standardmäßig wird davon ausgegangen, dass die neuen Koordinaten der BB zum gleichen Fahrzeug gehören

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])   # Berechnen des Abstandes vom alten zum neuen Mittelpunkt

                if dist < 100:  # wenn der Abstand einen Wert unterschreitet, wird angenommen, dass die neuen Koordinaten des Mittelpunktes zum gleichen Fahrezeug gehören
                    self.center_points[id] = (cx, cy)   # Die Mittelpunktkoordinaten werden bei der entsprechenden Fahrzeug-ID aktuallisiert
                    objects_bbs_ids.append([x, y, w, h, cx, cy, id])    # Die Liste mit den Fahrzeugdaten wird erweitert
                    obj_already_detected = True # Wert auf True gesetzt, damit folgender Code nicht durchlaufen wird

            # Neue Objekte erkennen
            if obj_already_detected is False:
                self.center_points[self.id_count] = (cx, cy)    # Die neuen Daten werden für die nächste Fahrzeug-ID abgespeichert
                objects_bbs_ids.append([x, y, w, h, cx, cy, self.id_count])
                self.id_count += 1  # Setzt den Counter für die Fahrzeuge nach oben für das nächste neue Fahrzeug

        new_center_points = {}  # Für neue Fahrzeuge
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, object_id = obj_bb_id # Liest die neue Fahrzeug-ID aus
            if not np.isin(object_id, self.car_in_out['id']):   # Überprüft, ob die ID schon abgelegt wurde
                self.car_in_out = np.append(self.car_in_out, np.array((object_id, 0, 0), dtype=self.car_in_out.dtype))  # Erweitert die Liste um die neue ID und dem Wert 0 für die Ein- und Ausfahrt
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()   # Die neuen Mittelpunktkoordinaten werden an das Dictionary übergeben
        return objects_bbs_ids  # Alle relevanten Variablen werden von der Funktion zurückgegeben

    def dist_to_line(self, line):   # Diese Funktion bestimmt den Abstand der Mittelpunkte zu einer Linie und überprüft, wann die Linie überquert wurde
        x1 = self.crossing_lines[line][0]   # Koordinaten der Linie entnehmen
        y1 = self.crossing_lines[line][1]
        x2 = self.crossing_lines[line][2]
        y2 = self.crossing_lines[line][3]

        for id in self.center_points:   # Aktualisiert den Mittelpunkt des Fahrzeugs
            x0, y0 = self.center_points[id]
            x0 += self.roi[0]   # Die Mittelpunktkoordinaten sind in roi definiert und müssen auf die Koordinaten von frame zurückgerechnet werden
            y0 += self.roi[1]

            dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)  # Berechnet den Abstand vom Punkt zur Linie

            if np.isin(0, self.car_in_out['in'][id]) and dist < 5:  # Überprüft, ob der Abstand zur Linie klein genug ist (nahe 0) und das Fahrzeug schon einmal die Linie überquert hat
                index = np.where(self.car_in_out['id'] == id)[0]    # Sucht die Stelle im Array, an der das Fahrzeug abgelegt wurde
                if len(index) > 0:  # Überprüft, ob das Fahrzeug im Array gefunden wurde
                    self.car_in_out['in'][index] = line + 1 # Setzt den Wert in der 'in'-Spalte an der entsprechenden Stelle auf line + 1 (damit es keine 0. Linie gibt)

            if np.isin(0, self.car_in_out['out'][id]) and dist < 5: # Gleich wie vorher nur für die Ausfahrt, also die 2. Linienüberquerung
                index = np.where(self.car_in_out['id'] == id)[0]
                if len(index) > 0:
                    if np.isin(line+1, self.car_in_out['in'][id]) or np.isin(0, self.car_in_out['in'][id]): # Stellt sicher, dass das Fahrzeug zuvor über eine andere Linie gefahren ist
                        None
                    else:
                        self.car_in_out['out'][index] = line + 1

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