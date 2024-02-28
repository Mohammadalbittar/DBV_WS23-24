import cv2
import numpy as np
from collections import Counter


# Funktion zum Erzeugen von befahrender Strecke
def roadExtract(video_path):
    # Erstellen von zwei Captures für Bildsubtraktion
    cap1 = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)


    # Variable zum Festlegen des ersten Frame
    first_frame = 100

    # Variablen zum Justieren des Ergebnisses
    frameDistance_var, iteration_var, threshold_var = 50, 10, 30

    # Festlegen des ersten Frame
    cap1.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, first_frame + frameDistance_var)

    # Array zum Abspreichern von den Ergebnissen der Subtraktion
    cumulative_result = []

    for _ in range(iteration_var):

        # Streaming
        _, main_capture = cap1.read()
        _, next_capture = cap2.read()

        # BGR2GRAY
        main_capture = cv2.cvtColor(main_capture, cv2.COLOR_BGR2GRAY)
        next_capture = cv2.cvtColor(next_capture, cv2.COLOR_BGR2GRAY)

        # Subtraktion des cap2 vom cap1
        frame_diff = cv2.absdiff(next_capture, main_capture)
        # frame_diff = next_capture - main_capture

        # Update the cumulative result
        _, binary_image = cv2.threshold(frame_diff, threshold_var, 255, cv2.THRESH_BINARY)

        cumulative_result.append(binary_image)

    # Anwenden der Funktion majority_pixel auf das resultierende Array
    befahreneStrecken = majority_pixel(cumulative_result)

    # Display Endergebnis
    cv2.imshow('befahrene Strecken', befahreneStrecken)
    # Fenster schließen bei Taste
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Funktion zum Vergleich Pixel an einer bestimmten Stelle über mehrere Frames
def majority_pixel(frames):
    # Ablesen der Höhe und Länge des ersten Frames aus dem gegebenen Array
    h, w = frames[0].shape
    # Erstellen einer 2D numpy Matrix
    majority_pixel = np.zeros((h, w), dtype=np.uint8)

    # Schleife über die Zeilen eines Frame
    for i in range(h):
        # Schleife über die Spalten eines Frame
        for j in range(w):
            # Ablesen des Pixel-Wertes an der Stelle i, j und Abspeichern in der pixel_values Array für alle frames 
            pixel_values = [frame[i, j] for frame in frames]
            # Anwenden der Counter-Funktion aus dem Modul "collections", um das meist vorkommende Pixel herauszufinden
            majority_value = Counter(pixel_values).most_common(1)[0][0]
            # Abspeichern der gefundene Pixel an der Stelle i, j in der 2D Matrix majority_pixel
            majority_pixel[i, j] = majority_value

    return majority_pixel
    

# Ausführen des Skript auf ein Video
roadExtract(r'resources\test_video.mp4')
