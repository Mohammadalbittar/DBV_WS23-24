# Analyse einer Kreuzung - Vergleich von Machine Learning zu Nicht-Machine-Learning Methoden

# Teammitglieder
Listen Sie hier noch einmal alle Teammitglieder mit Namen auf.

- Mohammad Albittar
- Noah Kasten
- Mostafa .. Nachname Ergänzen
- Jan Sbiegay

# Projektbeschreibung
Laut §9 der Prüfungsordnung ist die Prüfungsform die "Erstellung und Dokumentation von Rechnerprogrammen"

Die Erstellung und Dokumentation von Rechnerprogrammen umfasst in der Regel

1. die Beschreibung der Aufgabe und ihrer Abgrenzung,
2. die Erarbeitung theoretischer Voraussetzungen für die Bearbeitung der Aufgabe, insbesondere die Auswahl der geeigneten Methoden unter Einbeziehung und Auswertung einschlägiger Literatur,
3. die Formulierung der verwendeten Algorithmen in einer geeigneten Programmiersprache,
4. das Testen des Programms mit mehreren exemplarischen Datensätzen und das
Überprüfen der Ergebnisse auf ihre Richtigkeit,
5. die Programmdokumentation insbesondere mit Angabe der verwendeten Methoden,
des Ablaufplans, des Programmprotokolls (Quellenprogramm) und des Ergebnisprotokolls.

Bitte gehen Sie im Folgenden auf die oben genannten Punkte ein.

Die Ausarbeitung sollte sich im Idealfall wie ein wissenschaftlicher Artikel lesen, den man in einer Fachzeitschrift veröffentlichen könnte (und vielleicht machen wir dies auch, wenn Sie wollen).
Als Hilfestellung haben wir Ihnen typische Fragen in die jeweiligen Abschnitte geschrieben, welche natürlich innerhalb des Abschnittes möglichst beantwortet werden sollten. Die Angabe der Satzanzahl dient nur der Orientierung und muss nicht strikt befolgt werden. Ggf. für den besseren Lesefluss kann es auch sinnvoll sein Fragen zusammenzufassen oder wegzulassen. Dies ist gestattet, muss aber Sinn ergeben.

Als Sprache sind englisch oder deutsch erlaubt.

Anstatt in dieser ReadMe zu arbeiten, dürfen Sie auch ein PDF-Dokument erzeugen, z.B. mit LaTeX. In diesem Falle verlinken Sie bitte an dieser Stelle das entsprechende Dokument und legen alle zur Erstellung notwendigen Ressourcen in dem separaten Ordner *resources* ab.

## 1. Aufgabe und Abgrenzung
Dieser Abschnitt soll eine Art Überblick über den Rest der Arbeit bieten.

* Was ist das von Ihnen angegangene Problem. (1-2 Sätze)
* Warum ist dieses relevant? (1-2 Sätze)
* Was ist der wissenschaftliche Mehrwert Ihrer Arbeit? (1-3 Sätze)
* Kurzbeschreibung Ihres Ansatzes? (2-3 Sätze)
* Wie haben Sie Ihren Ansatz getestet? (1-2 Sätze)
* Was für nennenswerte Ergebnisse haben Sie erzielt? Bspw. 43% schneller, robuster als andere, etc.
* Überblick über ihr weiteres Vorgehen, welches unten beschrieben wird. (3-5 Sätze)

Mit diesem Projekt soll das Tracken vom Fahrzeugen in einer Kreuzung für die weitere Analyse der Straßensituation ermöglicht werden. Dazu werden einerseits Techniken mit traditionellen Verfahren der OpenCV-Bibliothek mit der Machine-Learning Methode YOLO verglichen. Dadurch soll herausgefunden werden, wie sich die Ergebnisse beider Ansätze unterscheiden und anschließed diskutiert werden, ob und wie sich diese gewinnbringend ergänzen können. (wissenschaftlicher mehrwert).
Für die Analyse der Kreuzung soll der Straßenverlauf erkannt werden, um die Verläufe des Verkehrsflusses anhand der gefahrenen Wege festzustellen. Dazu sollen an jeder Ein- bzw. Ausfahrt der Kreuzung automatisch Linien erkannt werden. Über die Mittelpunkte der getracktet Fahrzeuge können dann Abstände zu diesen Linien berechnet werden, die für das Zählen der Überquerten Linien verwendet werden. Mit diesen Daten kann anschließend der Verkehrsfluss, die Verkehrsdichte und die mittlere Geschwindigkeit in der Kreuzung erfasst werden.
(test)
(ergebnisse)

## 2. Theoretische Grundlagen und Methoden mit entsprechender Literaturangabe
- https://www.sciencedirect.com/science/article/pii/S0031320300000145#FIG3
- https://www.sciencedirect.com/science/article/pii/S0167865502001940?ref=pdf_download&fr=RR-2&rr=85d0c9f509504510#FIG1
- https://docs.ultralytics.com/
- Motion Extraction (https://www.youtube.com/watch?v=NSS6yAMZF78)

Klassische Ansätze versuchen bewegte Objekte über verschiedene Methoden zu Segmentieren und zu erkennen. Ausgehend von dieser Idee, versuchen wir bewegte Objekte von einem unbewegten Hintergrund zu trennen. 


* Worauf bauen Sie auf?
* Welche Literatur ist für Ihre Arbeit relevant? Dazu zählt nicht nur welche Sie verwendet haben, sondern auch, welche Arbeiten ähnliche Ziele wie Sie angehen.
* Worin unterscheiden sich diese von Ihrer Arbeit?

## 3. Formulierung des Vorgehens / Algorithmen


* <u>Ziele:</u> Bei jedem Ziel werden klassische Lösungsmöglichkeiten mit Machine Learining Lösungen verglichen
   * Fahrzeuge je Minute erkennen: 
      * Implementierung eines Algorithmus, der erkennt wieviele Fahrzeuge die Minute über die Kreuzung fahren

   * Verkehrsfluss Heatmap: 
      * Visualisierung des Verkehrsflusses der Kreuzung um eventuelle Muster und Intensitäten zu identifizieren

   * Mittlere Geschwindigkeit: 
      * Berechnung der mittleren Geschwindgkeit der erkannten Objekte
* Zunächst wurden mehrere Methoden für die Segmentierung untersucht, um die Objekte von dem Hintergrund zu trennen und eine Objekt-/ oder Hintergrundmaske zu erhalten
  
   * ERGEBNISSE VON JEDEM MIT DEN JEWEILIGEN FUNKTIONEN

      * Ergebniss
      * Motivation
      * Bewertung Ergebnis
   
   * Mohammad
      
   * Noah

Für das Erkennen und Verfolgen der Fahrzeuge mit OpenCV wurde unter Zuhilfenahme eines externen Ansatzes (siehe Theoretische Grundlagen) ein erster Ansatz entworfen. Dazu durchläuft das Programm eine Schleife über jeden Frame des zu analysierenden Videos. Über die CV-Funktion *findContours()* können dann ausgehendend von einer binären Maske bewegte Objekte im Video gefunden und mit der Funktion *boundingRect()* von einem Rechteck (Bounding-Box) umschlossen werden. Die weitere Objektverarbeitung erfolgt anschließend über die Klasse +Objecttracking()*, die in der *functions_n.py* definiert ist.

In dieser Klasse wurde die Funktion *add_new_vehicle()* erstellt. Über diese Funktion soll sichergestellt werden, dass die in jedem Frame erkannten Objekte zu einem Fahrzeug zugeordnet werden, um den Weg des Fahrzeugs zu verfolgen. Dazu werden zuerst für jeden Frame die Mittelpunkte der Bounding-Boxes berechnet. Anschließend wird der Abstand zwischen diesen Punkten bestimmt. Ist der Abstand klein genug, wird angenommen, dass das erkannte Objekt das gleiche Fahrzeug ist, andernfalls wird eine neue Fahrzeug-ID vergeben. Hierbei ist eine qualifizierte Erkennung durch die Maske grundlegend für die Funktionalität. Die Mittelpunkte, Bounding-Boxes und Fahrzeug-ID werden im Video visuell dargestellt (siehe MP_BB_ID).
![MP_BB_ID](resources/MP_BB_ID.png)
Um nun den Weg der Fahrzeuge für die Analyse des Verkehrsflusses in der Kreuzung aufzuzeichnen, wurde die Funktion *dist_to_line()* erstellt. Mit dieser Funktion kann der Abstand des Mittelpunktes zu einer Linie bestimmt werden. Die 4 Einmündungen in die Kreuzung stellen jeweils eine mögliche Linie, die ein Fahrzeug überqueren kann, dar. (siehe Veranschaulichung_Linien) Wenn ein Fahrzeug erstmalig über eine Linie fährt, wird diese Linie als Einfahrt in die Kreuzung in einem Array *„car_in_out“* gespeichert. Bei der zweiten Überquerung einer anderen Linie wird diese Linie als Ausfahrt aus der Kreuzung abgespeichert. Somit sollen alle Fahrtwege der Fahrzeuge identifiziert werden. 
![Veranschaulichung_Linien](resources/Veranschaulichung_Linien.png)
Aufgrund der unzureichenden eindeutigen Erkennung der Fahrzeuge wurde die Funktion *point_inside_polygon()* definiert. Hiermit soll verhindert werden, dass innerhalb der Kreuzung bzw. der Linien neue Fahrzeug-IDs vergeben werden, damit zu jedem Fahrzeug eine Ein- und eine Ausfahrt erkannt wird. In der Funktion wird dazu überprüft, ob sich der Mittelpunkt des Fahrzeugs innerhalb des Bereichs befindet. Ist dies der Fall, wird in der Funktion *add_new_vehicle()* keine neue ID vergeben (siehe Veranschaulichung_p_in_p). 
![Veranschaulichung_p_in_p](resources/Veranschaulichung_p_in_p.png)
Um den Rechenaufwand zu reduzieren, wird der gesamte Bildausschnitt auf eine „region of interest“ (roi) beschränkt. Für das Erstellen der roi wurde vorerst die Funktion *set_roi()* definiert. Diese ermöglicht, durch das Aufrufen der Funktion *mouse_callback()* und der Funktion *Image_from_Video()*, das Anklicken und Abspeichern von zwei Eckpunkten eines Rechtecks (siehe Veranschaulichung_set_roi). 
![Veranschaulichung_set_roi](resources/Veranschaulichung_set_roi.png)

   * Mostafa
   * Jan
   
* Anschließend wurden die Ansätze mit den visuell besten Ergebnissen ausgewählt und zu dem folgenden Hauptprogramm zusammengefügt: 
  
   * Ergebnisbild Hauptprogramm
   * Begründung für die Auswahl der Ergebnisse
   * Verlauf von Frame-Input zu Ergebnis-Output
   






* Beschreiben Sie erneut knapp das Ziel ihres Projekts (1-3 Sätze)
* Geben Sie zunächst einen Überblick über Ihr Programm, gerne auch mit entsprechender Visualisierung. Sie können Bilder zur besseren Erläuterung wie folgt einfügen: ![Mein Bild](resources/tiger.jpg) (3-5 Sätze)
* Motivieren Sie Ihren Ansatz! Warum denken Sie, dass dieser gut funktionieren wird? (2-4 Sätze)
* Was ist die genaue Eingabe- und Ausgabe Ihres Programms. (2-4 Sätze) Es ist völlig in Ordnung, wenn Sie sich auf den relevanten Teil konzentrieren und bspw. neben den Eingabebildern bereits perfekte (notfalls von Hand erzeugte) Vorder- und Hintergrundbilder erzeugen und als Eingabe nutzen. Erzeugen Sie sich die Eingabe, die Sie benötigen.
* Welche Annahmen verwenden Sie? Welche Annahmen verletzen Sie? Unter welchen Bedingungen sollte Ihr Ansatz definitiv funktionieren, wo wird er voraussichtlich Schwierigkeiten haben?
* Wie arbeitet ihr Programm? Welche Algorithmen wurden verwendet? Wie ist das Programm aufgebaut? Bitte konzentrieren Sie sich dabei auf die groben Konzepte, welche zum Verständnis notwendig sind und vermeiden Sie irrelevante Detailbeschreibungen.
* Wäre eine Tabelle für die Notationen hilfreich?
* Versuchen Sie eine sinnvolle Unterstruktur zu finden und vermeiden Sie alles in einen Abschnitt zu schreiben.
* Idealerweise wird jeder größere Schritt mit einem Beispiel validiert.
* Wenn Sie Bilder einbauen, stellen Sie sicher, dass die wichtigen Bereiche klar gezeigt werden (notfalls die wichtigen Bereiche vergrößern und separat darstellen)
* Erläutern Sie vor allem *warum* Sie etwas machen und nicht nur *was* sie machen.
* Erläutern Sie Ihren Ansatz vor allem zunächst theoretisch/verständlich und erst danach die Implementierung (falls diese überhaupt notwendig ist).


## 4. Evaluation

-  Getestet wurden mehrere Videos die über Youtube-Links oder als MP4 an das Programm übergeben wurden
   -  <u>Beispiellinks:</u>
      -  https://www.youtube.com/watch?v=2X27I6BAJcI
-  Hiermit sollte verdeutlich werden welche Stärken und Schwächen klassische Ansätze und derzeitige Machine Learning Ansätze haben
   -  <u>Klassisch:</u>
      -  Stärken und SChwächen einfügen
   -  <u>Machine Learning:</u>
      -  Stärken und Schwächen einfügen
   -  Abschließender Absatz: Was haben wir gezeigt, was konnten wir nicht zeigen
-  <u>Laufzeit:</u> 
   -  Klassisch FPS
   -  Machine Learning FPS (Loading + Inference + Postprocessing)
-  Limitierungen der Ansätze: 
   -  <u>Klassisch:</u>
   -  <u>Machine Learning:</u>
-  

​	



* Strukturieren Sie Ihre Ergebnisse!
* Auf welchen Datensätzen haben Sie ihr Programm getestet? (2-3 Sätze)
* Was wollten Sie damit zeigen? 
* Was konnten Sie damit zeigen? 
* Wie sind die Laufzeiten? (1-2 Sätze und optional Tabelle)
* Wie kann die Qualität der Ergebnisse gezeigt werden? (3-4 Sätze und optional Bilder und Tabellen)
* Wo sind Limitierungen? (2-3 Sätze und optional Bilder)
* Diskutieren Sie die Ergebnisse. 
* Schreiben Sie zuletzt eine Zusammenfassung, was Sie in diesem Projekt erreicht haben. (5-6 Sätze)
* Was hätte wie besser sein können? (1-2 Sätze)
* Wie könnte man den Ansatz erweitern oder weiter verbessern? (1-3 Sätze)
* Wer könnte von Ihrem Projekt profitieren? (1-4 Sätze)

## 5. Programmdokumentation
Dokumentieren Sie Ihren Quellcode leserlich und laden Sie alle relevanten Dateien und Datensätze in den "project"-Ordner hoch (sollte ihr Projekt mehr als 100MB verbrauchen, so legen Sie dieses bitte extern ab (bspw. GitHub, GDrive oder ähnliches) und verlinken dieses im "project"-Ordner.


# Arbeitsmatrix
Listen Sie hier alle relevanten Features als Liste mit sinnvoller Reihenfolge auf mit Angabe wer das Feature implementiert hat und zu welchem Anteil. Die Features sollen aus der Beschreibung des Projektes oben klar hervorgehen und nur die größeren Arbeitspakete umfassen (ca. 4-8 Features).
Bei starker Ungleicheit der Features behalten wir uns eine individuelle Notenvergabe vor.

Bspw.

- Systemarchitektur (Mohammad 0.0341 %, Berta 80%, Armin 20%)
- GUI (Carola 100%)
- ...
