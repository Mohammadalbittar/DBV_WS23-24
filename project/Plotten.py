import matplotlib.pyplot as plt
import numpy as np

def anzahlFahrzeugeProRichtung(input_array):
    # Initialisierung der Zähler für jede Richtung (Einfahrt und Ausfahrt)
    Richtung1_ein, Richtung2_ein, Richtung3_ein, Richtung4_ein, Richtung1_aus, Richtung2_aus, Richtung3_aus, Richtung4_aus = 0, 0, 0, 0, 0, 0, 0, 0
    
    # Durchlaufen der Eingabearrays und Zählen der Fahrzeuge pro Richtung
    for array in input_array:
        if array[1] == 1:
            Richtung1_ein += 1
        if array[1] == 2:
            Richtung2_ein += 1
        if array[1] == 3:
            Richtung3_ein += 1
        if array[1] == 4:
            Richtung4_ein += 1
        if array[2] == 1:
            Richtung1_aus += 1
        if array[2] == 2:
            Richtung2_aus += 1
        if array[2] == 3:
            Richtung3_aus += 1
        if array[2] == 4:
            Richtung4_aus += 1
        
    # Zusammenstellen der Gesamtanzahl pro Richtung (Einfahrt und Ausfahrt) in eine Liste
    GesamtAnzahlProRichtung = [[Richtung1_ein, Richtung2_ein, Richtung3_ein, Richtung4_ein], [Richtung1_aus, Richtung2_aus, Richtung3_aus, Richtung4_aus]]

    # Erstellen und Anzeigen des Balkendiagramms
    labels = ['Richtung 1', 'Richtung 2', 'Richtung 3', 'Richtung 4']  # Beschriftungen für die X-Achse
    x = range(len(labels))  # X-Positionen der Balken

    fig, ax = plt.subplots()  # Erstellen eines neuen Diagramms und einer Achse
    breite = 0.35  # Breite der Balken

    rects1 = ax.bar(x, GesamtAnzahlProRichtung[0], breite, label='Einfahrt')  # Erstellen der Balken für die Einfahrt
    rects2 = ax.bar([i + breite for i in x], GesamtAnzahlProRichtung[1], breite, label='Ausfahrt')  # Erstellen der Balken für die Ausfahrt

    ax.set_ylabel('Anzahl Fahrzeuge')  # Beschriftung der Y-Achse
    ax.set_title('Anzahl detektierte Fahrzeuge pro Richtung')  # Titel des Diagramms
    ax.set_xticks([i + breite / 2 for i in x])  # Positionen der X-Achsenbeschriftungen
    ax.set_xticklabels(labels)  # Beschriftungen der X-Achse
    ax.legend()  # Anzeige der Legende

    # Speichern des Diagramms als Bild
    plt.savefig('resources/diag_anzahlFahrzeugeProRichtung', bbox_inches='tight')

    # plt.show()  # Anzeigen des Diagramms


def anzahlFahrzeugeProMinute(videodauer_in_sec, CV_gesamt_erkannte_fahrzeuge, YOLO_gesamt_erkannte_fahrzeuge):
    # Labels für das Balkendiagramm
    labels = ['OpenCV', 'YOLO']
    videodauer_in_min = (int(videodauer_in_sec) / 60) + 1

    # Daten für das Balkendiagramm (Anzahl der erkannten Fahrzeuge pro Minute)
    erkannte_fahrzeuge_pro_minute = [CV_gesamt_erkannte_fahrzeuge / videodauer_in_min, YOLO_gesamt_erkannte_fahrzeuge / videodauer_in_min]

    # Balkendiagramm plotten
    plt.bar(labels, erkannte_fahrzeuge_pro_minute, color=['red', 'blue'])

    # Beschriftungen und Titel hinzufügen
    plt.xlabel('Erkennungsmethode')
    plt.ylabel('Anzahl der erkannten Fahrzeuge pro Minute')
    plt.title('Erkannte Fahrzeuge pro Erkennungsmethode')

    # Speichern des Diagramms als Bild
    plt.savefig('resources/diag_anzahlFahrzeugeProMinute', bbox_inches='tight')

    # # Aktiviere den interaktiven Modus
    # plt.ion()

    # # Zeige das Diagramm an
    # plt.show()

    # # Halte das Diagramm geöffnet, bis 'q' gedrückt wird
    # while True:
    #     # Warte auf Benutzereingabe
    #     key = input("Press 'q' to close the plot: ")
        
    #     # Überprüfe, ob die Eingabe 'q' ist
    #     if key.lower() == 'q':
    #         break
    # # Deaktiviere den interaktiven Modus
    # plt.ioff()

    # # Schließe das Diagramm
    # plt.close()