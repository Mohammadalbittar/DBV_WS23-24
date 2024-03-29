import cv2 as cv

#from project.GUI2 import *
from project.functions_n import *
from project.functions_j import *
from project.Plotten import *
from project.functions_m import *
import time
import matplotlib.pyplot as plt
import argparse


def main(uuser_title:str, path_to_file:str,  Live_testing:bool, user_live_output:bool, analysis_time:int):
    ######## Video Material ########

    path = path_to_file  # Videopfad
    url = 'https://www.youtube.com/watch?v=2X27I6BAJcI'  # URL für Testvideo




    ######## Initialisierung ########
    change_roi = False  # Wenn True, kann die roi mit der Funktion ot.set_roi angepasst werden
    auto_calc_roi = False  # Wenn True, wird die ROI automatisch berechnet
    # Wenn beide False, Standardwerte aus ot.roi[] verwendet

    ######## Initial Analysis ########
    cap = cv.VideoCapture(path)
    start_time = time.time()  # Startzeit des Videos
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    _, frame_one = cap.read()

    ######## Initialisierung Video Speichern als MP4########
    write_video = Live_testing  # Wenn True, wird das Video gespeichert
    output_video_time = analysis_time # Wieviele Sekunden des Videos sollen analysiert werden
    user_title = uuser_title # Titel des Videos
    if write_video:
        print('Video Output turned on')
        if user_title:
            timestamp = user_title
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

        write_video_path = f'user_results/output_{timestamp}.mp4'
        width_write = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_write = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_write = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out_cv_vid = cv.VideoWriter(write_video_path, fourcc, fps_write, (width_write, height_write*2))
        times_stat = [[], []]
        max_time = int(output_video_time*fps_write)
        current_frame = 0

    ######## Objecttracking ########
    ot = Objecttracking()    # ot als Objekt der Klasse Objecttracking definiert

    if change_roi:  # Wenn True, kann die roi mit der Funktion ot.set_roi angepasst werden
        ot.set_roi(ot.Imgage_from_Video(path, 100))

    if auto_calc_roi:

        # Berechne hintergrund image
        start_time_bg = time.time()*1000
        background_image, used_frames_bg = extract_background(cap, 500)
        # cv.imshow("Background",background_image)
        end_time_bg = time.time()*1000
        elapsed_time_bg = end_time_bg - start_time_bg

        # Berechne ROI
        # Finde stationäre Punkte
        start_time_roi = time.time()*1000
        points_stat, used_frames_Stat = find_Stats_point(cap,background_image,34,15000)
        #points = np.load("Points_Stationary.npy")
        #print(points)

        # Finde ROI Eckpunkten
        roi_eckpunkten = find_rois_points(background_image,points_stat)
        end_time_roi = time.time()*1000
        elapsed_time_roi = end_time_roi - start_time_roi
        #print(roi_eckpunkten)


    #Kernals für die Maske
    fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True)
    kernal_Op = np.ones((3,3),np.uint8)  # Öffnung
    kernal_Cl = np.ones((8,8),np.uint8)  # Schließung
    kernal_e = np.ones((4,4),np.uint8)  # Erodieren

    yolo_regio = yolo_region_count([(460, 370), (520, 520), (1150, 460), (940, 345)])

    ######## Anwendungsteil ########
    cap = cv.VideoCapture(path)
    start_time = time.time()  # Startzeit des Videos
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    _, frame_one = cap.read()

    while True:
        ret, frame = cap.read() # Frame einlesen
        if not ret:
            break
        frame_y = frame.copy()

        if auto_calc_roi: # Übergibt die entscheidenden Punkte aus der automatischen ROI Berechnung an die roi für ot
            ot.roi[0] = roi_eckpunkten[3][0] # x1
            ot.roi[1] = roi_eckpunkten[3][1] # y1
            ot.roi[2] = roi_eckpunkten[0][0] # x2
            ot.roi[3] = roi_eckpunkten[0][1] # y2

        roi = frame[ot.roi[1]:ot.roi[3], ot.roi[0]: ot.roi[2]] # Region_of_interest Format: y1, y2 : x1, x2
        #cv2.imshow('ROI', roi)

        ## 1. Zeitmessung Anfang
        start_time_cv = time.time()*1000

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

        end_time_cv = time.time()*1000  # Endzeit der Zeitmessung
        elapsed_time_cv = end_time_cv - start_time_cv # Dauer, die die Bildverarbeitung benötigt hat
        if write_video:
            times_stat[0].append(elapsed_time_cv)
        frame = add_text_to_frame(frame, f'{elapsed_time_cv:.2f} ms/frame')

        ###### YOLO ######
        start_time_yolo = time.time() *1000 # Startzeit der Zeitmessung
        frame_yolo, ins, out = yolo_regio(frame_y)
        end_time_yolo = time.time()*1000  # Endzeit der Zeitmessung
        elapsed_time_yolo = end_time_yolo - start_time_yolo
        if write_video:
            times_stat[1].append(elapsed_time_yolo)
        frame_yolo = add_text_to_frame(frame_yolo, f'{elapsed_time_yolo:.2f} ms/frame')

        ##### Ausgabe von Bildern #####
        frame = video_tiling_mixed(frame, frame_yolo, width, height)

        if user_live_output:
            cv.imshow('Frame', frame)
        #cv.imshow('Maske', e_img)
        #cv.imshow('Region of Interest', roi)
        #cv.imshow('Frame', frame)
        #cv.imshow('Frame YOLO', frame_yolo)
        
        key = cv.waitKey(30)
        if key == 27:   # Durch Drücken der ESC-Taste wird das Programm geschlossen
            break

        if write_video:
            frame = cv.resize(frame, (width_write, height_write*2))
            out_cv_vid.write(frame)  # Schreibt das Bild in das Video
            current_frame +=1
            print(current_frame, '/', max_time)
            if max_time == current_frame:
                break


    cap.release()
    cv.destroyAllWindows()

    end_time = time.time()  # Endzeit des Videos
    elapsed_time = end_time - start_time  # Dauer, die das Video abgespielt wurde


    ###### Zeitmessung Graph Speichern######
    if write_video:
        x_values = range(1, len(times_stat[0])+1)
        plt.plot(x_values, times_stat[0], label='OpenCV')
        plt.plot(x_values, times_stat[1], label='YOLO')
        plt.ylim(0, 150)
        plt.grid()
        plt.xlabel('Frame')
        plt.ylabel('Time [ms]')
        plt.legend()
        if user_title:
            graph_output_title = f'user_results/graph_{user_title}.png'
        else:
            graph_output_title = f'user_results/graph_{timestamp}.png'
        plt.savefig(graph_output_title)
        out_cv_vid.release()
        print(f'Video saved as {write_video_path}')


    # Daten auswerten
    anzahlFahrzeugeProRichtung(ot.car_in_out)
    anzahlFahrzeugeProMinute(elapsed_time, len(ot.car_in_out), ins)

# Main
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Description of your script.")
    #parser.add_argument("param_str1", type=str, help="Name of User Output")
    #parser.add_argument("param_str2", type=str, help="path to video file")
    #parser.add_argument("param_bool1", type=lambda x: x.lower() == 'true',
                        #help="Output ('True' or 'False')")
    #parser.add_argument("param_bool2", type=lambda x: x.lower() == 'true',
                        #help="Live View ('True' or 'False')")
    #parser.add_argument("param_int", type=int, help="how many seconds of video to analyse")

    #args = parser.parse_args()

    #main(args.param_str1, args.param_str2, args.param_bool1, args.param_bool2, args.param_int)
    main('test', r'C:\Users\mosta\PycharmProjects\DBV_WS23-24\resources\video2.mp4', True, True, 10000)