import cv2
import numpy as np

def motion_extraction(frame1, frame2):    #  Function von Jan         
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)          # Frame 2 in Graustufen konvertieren
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)          # Frame 1 in Graustufen konvertieren
    result = cv2.addWeighted(frame1, 0.5, cv2.bitwise_not(frame2), 0.5, 0)          # Gewichtete Addition der beiden Frames
    result = result.astype(np.float32) / 255          # Konvertierung des Ergebnisframes in den Bereich von 0 bis 1
    result = abs(result - 0.5)          # Betragsdifferenz zwischen dem Ergebnisframe und 0.5
    _, result = cv2.threshold(result, 0.35, 1, cv2.THRESH_BINARY)          # Binarisierung des Ergebnisframes
    result = cv2.dilate(result, None, iterations=10)          # Dilatation des binarisierten Ergebnisframes
    return result

def process_video(video_path):
    cap1 = cv2.VideoCapture(video_path)          # Video-Capture-Objekt für das Hauptvideo erstellen
    cap2 = cv2.VideoCapture(video_path)          # Video-Capture-Objekt für das sekundäre Video erstellen

    cap1.set(cv2.CAP_PROP_POS_FRAMES, 1800)          # Position des Hauptvideos festlegen
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 1850)          # Position des sekundären Videos festlegen

    one_time = True          # einmalige Ausführung
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))          # Gesamtzahl der Frames im Video erhalten

    while True:
        ret1, main_capture = cap1.read()          # Frame aus dem Hauptvideo lesen
        ret2, sec_capture = cap2.read()           # Frame aus dem sekundären Video lesen
        
        if not ret1 or not ret2:
            print("Error: Leerer Frame gefunden.")
            break

        if one_time:
            cumulative_result = np.zeros_like(main_capture[:, :, 0], dtype=np.float32)          # Kumulatives Ergebnis initialisieren
            _, frame_one = cap1.read()          # Ein Frame lesen
            one_time = False          # Flag aktualisieren
        
        binary_image = motion_extraction(main_capture, sec_capture)          # Bewegungsextraktion zwischen den Frames
        
        cumulative_result += binary_image          # Kumulatives Ergebnis aktualisieren
        cumulative_result = cv2.dilate(cumulative_result, None, iterations=15)          # Dilatation des kumulativen Ergebnisses
        cumulative_result = cv2.erode(cumulative_result, None, iterations=15)          # Erosion des kumulativen Ergebnisses
    return cumulative_result.astype(np.uint8)          # Rückgabe des kumulativen Ergebnisses als uint8-Array


# Funktion process_video aufrufen und Schwellwert anwenden
_, process_and_threshold = cv2.threshold(process_video(r'resources\test_video.mp4'), 1, 255, cv2.THRESH_BINARY)

# Kanten durch Canny-Detektion finden
edges = cv2.Canny(process_and_threshold, 100, 120)
edges = cv2.dilate(edges, None, iterations=10)          # Dilatation der Kanten
edges = cv2.erode(edges, None, iterations=7)            # Erosion der Kanten

# Linien durch Hough-Transformation finden
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 400, 800, 1)
print(lines, 'Anzahl:', len(lines))
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(edges, (x1, y1), (x2, y2), (180, 34, 255), 15)          # Linien auf das Kantenbild zeichnen

cv2.imshow('edges', edges)          # Kantenbild anzeigen
cv2.waitKey(0)          # Auf eine Benutzereingabe warten
