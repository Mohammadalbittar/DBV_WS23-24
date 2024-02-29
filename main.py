from project.GUI2 import *
from project.functions_n import *
from project.functions_j import *



def main():
    ######## URL für Videos ########


    url = 'https://www.youtube.com/watch?v=2X27I6BAJcI'  # URL für Testvideo

    ######## Initialisierung ########
    path = r'C:\Noah\Studium Lokal\Master\DBV_Abschlussprojekt\TestVideo1.mp4'  # Videopfad

    cap = get_livestream(url)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    length  = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    _, frame_one = cap.read()

    ot = Objecttracking()    # ot als Objekt der Klasse Objecttracking definiert

    #Kernals für die Maske
    fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)
    kernal_Op = np.ones((3,3),np.uint8)  # Öffnung
    kernal_Cl = np.ones((8,8),np.uint8)  # Schließung
    kernal_e = np.ones((4,4),np.uint8)  # Erodieren



    ######## Anwendungsteil ########
    cap = cv.VideoCapture(path)
    start_time = time.time()  # Startzeit des Videos

    while True:
        ret, frame = cap.read() # Frame einlesen
        roi = frame[ot.roi[1]:ot.roi[3], ot.roi[0]: ot.roi[2]] # Region_of_interest Format: y1, y2 : x1, x2

        #Erstellen der Maske
        fgmask = fgbg.apply(roi)    # Vordergrund vom Hintergrund trennen
        ret, imBin = cv.threshold(fgmask, 254, 255, cv.THRESH_BINARY)   # Binäre Maske erstellen
        mask1 = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernal_Op) # Rauschen entfernen, Lücken schließen
        mask2 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, kernal_Cl)    # Löcher füllen
        mask = cv.erode(mask2, kernal_e)   # Verkleinern

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   # zusammenhängende Konturen auf der Maske erkennen
        bounding_boxes = [] # Liste in der Rechtecke um die Objekte gespeichert werden sollen

        for cnt in contours:
            area = cv.contourArea(cnt)  # Berechnet die Fläche der erkannten Kontur
            if area > 600:  # Zeichnet ein Rechteck (BB) um die Kontur, wenn die Fläche einen Schwellwert überschreitet
                x, y, w, h = cv.boundingRect(cnt)
                bounding_boxes.append([x, y, w, h]) # Koordinaten der BB in der Liste speichern

        boxes_ids = ot.add_new_vehicle(bounding_boxes)  # Trackingfunktion auf die BB anwenden
        for box_id in boxes_ids:    # Aktualisiert die gezeichneten BB, Mittelpunkte und Fahrzeug-IDs
            x, y, w, h, cx, cy, id = box_id
            cv.circle(roi, (cx,cy), 5, (0, 0, 255), -1)
            cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

            ot.dist_to_line(0)  # links
            ot.dist_to_line(1)  # unten
            ot.dist_to_line(2)  # rechts
            ot.dist_to_line(3)  # oben

            print("")
        print(ot.car_in_out)

        #data_plot.plotData(ot.car_in_out, cap)

        # Einzeichnen der Linien, an denen die Überquerung gezählt wird
        cv.line(frame, (ot.crossing_lines[0][0], ot.crossing_lines[0][1]), (ot.crossing_lines[0][2], ot.crossing_lines[0][3]), (0, 0, 255), 2)  # links
        cv.line(frame, (ot.crossing_lines[1][0], ot.crossing_lines[1][1]), (ot.crossing_lines[1][2], ot.crossing_lines[1][3]), (0, 0, 255), 2)  # unten
        cv.line(frame, (ot.crossing_lines[2][0], ot.crossing_lines[2][1]), (ot.crossing_lines[2][2], ot.crossing_lines[2][3]), (0, 0, 255), 2)  # rechts
        cv.line(frame, (ot.crossing_lines[3][0], ot.crossing_lines[3][1]), (ot.crossing_lines[3][2], ot.crossing_lines[3][3]), (0, 0, 255), 2)  # oben

        #cv.imshow('Maske', e_img)
        #cv.imshow('Region of Interest', roi)
        cv.imshow('Frame', frame)

        key = cv.waitKey(30)
        if key == 27:   # Durch Drücken der ESC-Taste wird das Programm geschlossen
            break

    end_time = time.time()  # Endzeit des Videos
    elapsed_time = end_time - start_time  # Dauer, die das Video abgespielt wurde

    cap.release()
    cv.destroyAllWindows()

# Main
if __name__ == "__main__":
    main()
    #root = tk.Tk()
    #gui = GUI(root)
    #root.mainloop()