import cv2
import numpy as np

def process_video(video_path):
    # Video-Capture-Objekte für Hauptframe und nächstes Frame erstellen
    cap1 = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path)

    # Erstes Frame festlegen
    first_frame = 100

    # Position der Videoframes für die beiden Capture-Objekte setzen
    cap1.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, first_frame + 50)

    # Einmalige Initialisierung für kumulatives Ergebnis
    one_time = True

    while True:
        # Frames von den beiden Capture-Objekten lesen
        _, main_capture = cap1.read()
        _, next_capture = cap2.read()

        # Hauptframe und nächster Frame in Graustufen konvertieren
        main_capture = cv2.cvtColor(main_capture, cv2.COLOR_BGR2GRAY)
        next_capture = cv2.cvtColor(next_capture, cv2.COLOR_BGR2GRAY)

        if one_time:
            # Kumulatives Ergebnis initialisieren
            cumulative_result = np.zeros_like(main_capture, dtype=np.float32)
            one_time = False

        # Differenz zwischen den Frames berechnen
        frame_diff = cv2.absdiff(next_capture, main_capture)

        # Bild binarisieren
        _, binary_image = cv2.threshold(frame_diff, 1, 255, cv2.THRESH_BINARY)

        # Kumulatives Ergebnis aktualisieren
        cumulative_result += binary_image

        # Frames anzeigen
        cv2.imshow('Input Video', main_capture)
        cv2.imshow('Kumulative Differenz', cumulative_result.astype(np.uint8))

        # Auf Tastatureingabe warten (Drücken von 'q' zum Beenden)
        if cv2.waitKey(1) == ord('q'):
            break

    # Video-Capture-Objekte freigeben und Fenster schließen
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Beispielaufruf der Funktion mit dem Pfad zum Video
process_video(r'resources\test_video.mp4')
