import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


# Erstellen einer Figur mit den entsprechenden Axen
fig, ax = plt.subplots()
x, y = [0], [0]

# Initialisierung eines interktiven Plottes
plt.ion()

# Erstellen einer dynamischen Linie:
dynLinie, = ax.plot([], [], 'b-')

# Bezeichnungen
ax.set_title("detektierte Fahrzeuge über Zeit (opencv)")
ax.set_xlabel("Zeit")
ax.set_ylabel("Anzahl detektierte Fahrzeuge")

# Funktion zum Plotten der Daten
def plotData(array = None, cap = None):

    # Ablesen des Zeitfortschrittes der Funktion cv.VideoCapture() in Sekunden
    zeitFortschritt = cap.get(cv.CAP_PROP_POS_MSEC)/1000
    print('vergangene Zeit:', zeitFortschritt, 'Anzahl detektierter Fahrzeuge: ', len(array))

    # Neue Werte der dynamischen Liste hinzufügen
    x.append(zeitFortschritt)
    y.append(len(array))

    # Aktualisierung der Werte der Axen
    dynLinie.set_data(x, y)

    # Aktualisierung der Größe der Figur
    ax.relim()
    ax.autoscale_view()

    # Plotten der aktualisierten Werte
    plt.draw()

    # Pause, um das Plot zu aktualisieren
    plt.pause(0.001)

    return


