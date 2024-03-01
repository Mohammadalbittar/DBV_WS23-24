# Analyse einer Kreuzung - Vergleich von Machine Learning zu Nicht-Machine-Learning Methoden

# Teammitglieder
Listen Sie hier noch einmal alle Teammitglieder mit Namen auf.

- Mohammad Albittar
- Noah Kasten
- Mostafa Mahmoud
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
Mit diesem Projekt soll das Tracken vom Fahrzeugen in einer Kreuzung für die weitere Analyse der Straßensituation ermöglicht werden. Dazu werden einerseits Techniken mit traditionellen Verfahren der OpenCV-Bibliothek mit der Machine-Learning Methode YOLO verglichen. Dadurch soll herausgefunden werden, wie sich die Ergebnisse beider Ansätze unterscheiden und anschließed diskutiert werden, ob und wie sich diese gewinnbringend ergänzen können. Die gegenseitige Ergänzung könnte später für die Optimierung von bisherigen Implementierung für die Analyse von Verkehrsfluss genutzt werden um z.B. diesen bei der Stadtplanung zu berücksichtigen. Mit dieser Arbeit soll zunächst getestet werden wie sich mögliche Implementierungen in Qualität und Performance unterscheiden. Die Umstetzung soll anschließend mit verschiedenen Videos auf verschiedenen Systemen getestet werden. Als Ergebnis konnten wir die Performance Unterschiede von unserem OpenCV Ansatz mit dem von YoloV8 im direkten Vergleich darstellen und Stärken und Schwächen von beiden erläutern. 

Für die Analyse der Kreuzung soll zunächst versucht werden den Straßenverlauf zu erkennen. Dieser soll anhand der Verläufe des Verkehrsflusses der gefahrenen Wege erkannt werden. Dazu soll versucht werden an jeder Ein- bzw. Ausfahrt der Kreuzung automatisch Linien zu erkennen. Über die Mittelpunkte der getracktet Fahrzeuge können dann Abstände zu diesen Linien berechnet werden, die für das Zählen der überquerten Linien verwendet werden. Mit diesen Daten kann anschließend der Verkehrsfluss, die Verkehrsdichte und die mittlere Geschwindigkeit in der Kreuzung erfasst werden.

Auch soll versucht werden, eine **R**egion**O**f**I**nterest und ein Kreuzungsbereich über stehende Fahrzeuge an der Ampelmarkierung zu ermitteln. Dies soll auf Grundlage der langzeitbetrachtung der Bilderserie geschehen, um die stehenden Fahrzeuge zu ermitteln. 

Auch soll eine Umsetzung von YoloV8 implementiert werden, um die Ergebnisse direkt vergleichen zu können. 

Anschließend sollen die Ergebnisse von Verkehrsanalyse/Verkehrsfluss, die Video- und Performanceergebnisse in dem Ergebnisordner "user_results" ausgegeben werden. 

## 2. Theoretische Grundlagen und Methoden mit entsprechender Literaturangabe
- https://www.sciencedirect.com/science/article/pii/S0031320300000145#FIG3 [1]
- https://www.sciencedirect.com/science/article/pii/S0167865502001940?ref=pdf_download&fr=RR-2&rr=85d0c9f509504510#FIG1 [2]
- https://docs.ultralytics.com/
- Motion Extraction (https://www.youtube.com/watch?v=NSS6yAMZF78)
- Grundlage OpenCV Objekterkennung (https://youtu.be/O3b8lVF93jU)
- Grundlage OpenCV Objekterkennung (https://github.com/dhananjaymenon/SpeedRadar-OpenCV-/blob/a8934ab99dd8dd6f891f3b60faa03806751c5944/tracker2.py)
- Grundlage OpenCV Objekterkennung (https://github.com/dhananjaymenon/SpeedRadar-OpenCV-/blob/a8934ab99dd8dd6f891f3b60faa03806751c5944/SpeedRadar2.py)
- Bachround Removal (https://www.youtube.com/watch?v=fEi63QB81ek)
- OpenCV Doku Seite
- ChatGPT
- Hilfestellung für Machine Learning (https://supervision.roboflow.com/how_to/track_objects/#__tabbed_2_1)


Klassische Ansätze [1] [2] versuchen bewegte Objekte über verschiedene Methoden wie z.B. Bewegung zu Segmentieren und zu erkennen. Ausgehend von dieser Idee, versuchen wir bewegte Objekte von einem unbewegten Hintergrund zu trennen. Relevant für unsere Arbeit war vorallem die Dokumentation von OpenCV und die Video Tutorials sowie die Dokumentation von YoloV8. Im Gegensatz zu den zu den genannten Quellen [1] [2] werden fertige Funktionen von OpenCV als Grundlage genutzt. 

## 3. Formulierung des Vorgehens / Algorithmen


* <u>Ziele:</u> Bei jedem Ziel werden klassische Lösungsmöglichkeiten mit Machine Learining Lösungen verglichen
   * Fahrzeuge je Minute erkennen: 
      * Implementierung eines Algorithmus, der erkennt wieviele Fahrzeuge die Minute über die Kreuzung fahren

   * Verkehrsfluss Heatmap: 
      * Visualisierung des Verkehrsflusses der Kreuzung um eventuelle Muster und Intensitäten zu identifizieren

   * Mittlere Geschwindigkeit: 
      * Berechnung der mittleren Geschwindgkeit der erkannten Objekte
* Zunächst wurden mehrere Methoden für die Segmentierung untersucht, um die Objekte von dem Hintergrund zu trennen und eine Objekt-/ oder Hintergrundmaske zu erhalten
  
   * Noah

Für das Erkennen und Verfolgen der Fahrzeuge mit OpenCV wurde unter Zuhilfenahme eines externen Ansatzes (siehe Theoretische Grundlagen) ein erster Ansatz entworfen. Dazu durchläuft das Programm eine Schleife über jeden Frame des zu analysierenden Videos. Über die CV-Funktion *findContours()* können dann ausgehendend von einer binären Maske bewegte Objekte im Video gefunden und mit der Funktion *boundingRect()* von einem Rechteck (Bounding-Box) umschlossen werden. Die weitere Objektverarbeitung erfolgt anschließend über die Klasse *Objecttracking()*, die in der *functions_n.py* definiert ist. Die Verwendung einer Klasse ist vorteilhaft, da für verschiedene Quellen andere Parameter benötigt werden und somit einfach ein neues Klassenobjekt angelegt werden kann.

In dieser Klasse wurde die Funktion *add_new_vehicle()* erstellt. Über diese Funktion soll sichergestellt werden, dass die in jedem Frame erkannten Objekte zu einem Fahrzeug zugeordnet werden, um den Weg des Fahrzeugs zu verfolgen. Dazu werden zuerst für jeden Frame die Mittelpunkte der Bounding-Boxes berechnet. Anschließend wird der Abstand zwischen diesen Punkten bestimmt. Ist der Abstand klein genug, wird angenommen, dass das erkannte Objekt das gleiche Fahrzeug ist, andernfalls wird eine neue Fahrzeug-ID vergeben. Hierbei ist eine qualifizierte Erkennung durch die Maske grundlegend für die Funktionalität. Die Mittelpunkte, Bounding-Boxes und Fahrzeug-ID werden im Video visuell dargestellt (siehe MP_BB_ID).
<center><img src="resources/MP_BB_ID.png" alt="MP_BB_ID" style="zoom:150%;" /></center>

Um nun den Weg der Fahrzeuge für die Analyse des Verkehrsflusses in der Kreuzung aufzuzeichnen, wurde die Funktion *dist_to_line()* erstellt. Mit dieser Funktion kann der Abstand des Mittelpunktes zu einer Linie bestimmt werden. Die 4 Einmündungen in die Kreuzung stellen jeweils eine mögliche Linie, die ein Fahrzeug überqueren kann, dar (siehe Veranschaulichung_Linien). Wenn ein Fahrzeug erstmalig über eine Linie fährt, wird diese Linie als Einfahrt in die Kreuzung in einem Array *„car_in_out“* gespeichert. Bei der zweiten Überquerung einer anderen Linie wird diese Linie als Ausfahrt aus der Kreuzung abgespeichert. Somit sollen alle Fahrtwege der Fahrzeuge identifiziert werden, sofern das Verfolgte Objekt nicht "verloren" geht. 

<center><img src="resources/Veranschaulichung_Linien.png" alt="Veranschaulichung_Linien" /></center>

Aufgrund der unzureichenden eindeutigen Erkennung der Fahrzeuge wurde die Funktion *point_inside_polygon()* definiert. Hiermit soll verhindert werden, dass innerhalb der Kreuzung bzw. der Linien neue Fahrzeug-IDs vergeben werden, damit zu jedem Fahrzeug eine Ein- und eine Ausfahrt erkannt wird. In der Funktion wird dazu überprüft, ob sich der Mittelpunkt des Fahrzeugs innerhalb des Bereichs befindet. Ist dies der Fall, wird in der Funktion *add_new_vehicle()* keine neue ID vergeben (siehe Veranschaulichung_p_in_p). 
<center><img src="resources/Veranschaulichung_p_in_p.png" alt="Veranschaulichung_p_in_p" /></center>
Um den Rechenaufwand zu reduzieren, wird der gesamte Bildausschnitt auf eine „region of interest“ (roi) beschränkt. Für das Erstellen der roi wurde vorerst die Funktion *set_roi()* definiert. Diese ermöglicht, durch das Aufrufen der Funktion *mouse_callback()* und der Funktion *Image_from_Video()*, das manuelle Anklicken und Abspeichern von zwei Eckpunkten eines Rechtecks (siehe Veranschaulichung_set_roi). 
<center><img src="resources/Veranschaulichung_set_roi.png" alt="Veranschaulichung_set_roi" /></center>

   * Mostafa

Neben dem Nutzer definierten ROI, wird auch eine ROI automatisch bestimmt. Dabei wurde eine Methode zum Extrahieren von den vier Eckpunkten des ROIs entwickelt. Dafür wurde 3 Funktionen geschrieben, extract_background(), find_Stats_point() und find_rois_points().

Die Grundidee zum Berechnen der ROI Eckpunkte ist das Erkennen von an der Ampel stillstehenden Autos und dadurch Punktwolken zu erzeugen. Aus den Punktwolken können dann die Schwerpunkte bestimmt werden. Die Schwerpunkten der Punktwolken sollen bei einer typisch Ampelkreuzung auf den 4 Seiten des ROIs Rechtecks liegen.

Um die stehenden Autos zu erkennen, muss eine Maske erstellt werden, die nur stillstehende Autos erkennt. Für die Erzeugung der Maske werden zwei andere Masken miteinander verglichen, diese sind:
* Maske für bewegende Autos
* Maske für bewegende und still stehende Autos

Die erste Maske lässt sich durch hauptsächlich 2 OpenCV Funktionen berechnen, createBackgroundSubtractorKNN() und createBackgroundSubtractorMOG2(). Lediglich createBackgroundSubtractorKNN() wird verwendet, da es durch Experiementieren zu besseren Ergebnisse geführt hat. Die Berechung der zweiten Maske erfolgt durch den Vergleich zwischen dem Hintergrundbild und dem aktuellen Frame. Die Erzeugung des Hintergrundbilds geschieht in der selbst definierten Funktion, extract_background(). In der Funktion wird das Video Capture gesamplet. Von den gesampleten Frames wird der Median berechnet, was dann  den Hintergrund ergibt. Die Ergebnisse dieser Funktion werden dann für die folgende Funktion, find_Stats_point(), verwendet. Diese Funktion ergibt als Output Punkt, wo im Video Fahrzeug stillstanden. In der Funktion wird zuerst eine Differenzmaske erstellt. Die Maske ist die Differenz zwischen dem zuvor erzeugten Hintergrund Bild und dem aktuellen Frame. Dadurch werden alle Autos erkannt.
<center><img src="resources/Differenz_Maske.PNG" alt="Differenz Maske" /></center>

Von dieser Maske werden nun die bewegenden Autos subtrahiert. Durch createBackgroundSubtractorKNN() lassen sich die bewegenden Autos erkennen. Die erkannten Punkte werden in der Diff Maske identifiziert und auf Null gesetzt. Damit erhält man das folgende:
<center><img src="resources/Auto_KNN.png" alt="Auto KNN Subtractor" /></center>
<center><img src="resources/Auto_geloescht.png" alt="Bewegende Auto subtrahiert" /></center>

Zum Erkennen von Autos wird die OpenCV Funktionen verwendet, findContours(), und wird die Fläche berechnet. Nur die Fläche über eine bestimmten Threshold, was empirisch ermittelt wurde, werden als Objekt erkannt. Der Grund dafür ist, das kleine unerwüschte Störungen nicht erkannt werden. Um die Contours wird ein Rechteck gezeichnet und der Mittelpunkt des Reckteckes berechnet, gezeichnet und gespeichert.
<center><img src="resources/erkannten_Still_stehnde_Autos.PNG" alt="erkannten stehende Autos" /></center>

Von dem gespeicherten Mittelpunkten werden in find_rois_points() die Schwerpunkten der Punktwolken gerechnet. Hier wird dafür der K-means-clustering Algorithmus verwendet.
<center><img src="resources/Punktwole%20und%20Centeriods.PNG" alt="Punktwolken und Clusters Schwerpunkten" /></center>

Die Centerroids sollen auf den vier Seite des ROIs Rechtecks liegen. Linien werden aus dem vier Centerriods gezeichnet. Die Schnittpunkten der Linien entsprechend die Eckpunkten des ROIs und werden als Output ausgegeben. 
<center><img src="resources/berechnetes%20ROI.PNG" alt="berechnetes ROI" /></center>

   * Jan
Vergleich der Verschiedenen Methoden für die Hintergrundsegmentierung. Hier wurden die Verschiedenen Umsetzungen von dem OpenCV internen BackgroundSubtractor getestet, MOG2, KNN, CNT und GMG. Die Verschiedenen Umsetzungen lieferten gemischte Ergebnisse, da gelegentlich der Output flackerte oder wenn Autos über lägere Zeit still standen diese als Hintergrund klassifiziert wurden und somit aus der Ausgabe verschwanden. Dies hat den Hintergrund, dass die Umsetzungen des BackgroundSubtractor mit einer History arbeiten und hierbei jedem Pixel einen, auf Wahrscheinlichkeit über die Bildhistorie betrachtet, basierenden Wert zuweisen.
<center><img src="resources/BackroundSub.png" alt="BackroundSub" /></center>
Als nächstes wurde eine weitere Methode für die Motion Extraction aus Video getestet. Diese Methode basiert auf der Idee der Quelle [Quelle]. Hierbei werden zwei aufeinander folgende Frames miteinander verrechnet. Bewegt sich ein Objekt im Bild, entsteht eine Kante die um die Pixelwertänderung vom Mittelwert des Maximalwertes des Datentyps (Beispiel: Datentyp uint8 -> Mittelwert 128) abweicht. Subtrahiert man nun den Mittelwert vom Bild und bildet den Absoluter. so erhält man ein schwarzes Bild wo nur durch Bewegung erzeugte Kanten vom 0 Wert abweichen. Durch das zusätzliche Anwenden des OpenCV Threshholding Algorithmus auf das Ergebnis, kann Rauschen und minimale Änderungen rausgefiltert werden und man erhält eine binäre Masse mit "Bewegung" =1 und "Unbewegt" = 0. 
<center><img src="resources/Motion.png" alt="Motion" /></center>
Anschließend wurde noch eine Umsetzung des Lukas Kanade Algorithmus für Dense Optical Flow von OpenCV getestet. Ziel war es, anhand eines Vektorfeldes einzelne bewegte Objekte im Bild zu segmentieren, und sich überlagernde Objekte anhand des Algorithmus zu differenzieren. Die Umsetzung hiervon war sehr rechenintensiv und langsam, weshalb für eine spätere eventuelle, alternative Verwendung unter anderem nur die Magnitude und der Winkel als Rückgabeparameter implementiert wurden. 
<center><img src="resources/Lukas_Kanade_1.png" alt="Lukas_Kanade_1" /></center>

Um die unterschiedlichen Binärmasken aus den Ergebnissen der vorhergehenden FUnktionen zu verbessern, wurden unterschieliche Clustering Methoden betrachtet. Diese waren aber aufgrund einer sehr hohen Laufzeit, bedingt doch die große Menge an Punkten in den Punktwolken bei hoher Auflösung, nicht sinnvoll anwendbar oder brauchten zur initialisierung eine Anzahl der gewünschten Cluster, was in diesem Anwendungsfall nicht sinnvoll ist. 
Für die Umsetzung mit Machine Learning wurde YoloV8 ausgewählt, da dieses vergleichsweise sehr gut Dokumentiert war und die gegebenen Modelle ausreichend trainiert waren, sodass diese auf Anhieb funktionieren konnten. Hierfür wurden auf der Grundlage von zwei Online Beispielen zwei Umsetzungen getestet, explizit wurden die Modi Predict und Track getestet, jedoch waren für unseren Anwendungsfall hier keine gravierenden Unterschiede ersichtlich.<center><img src="resources/yolo_region.png" /></center>

   * Mohammad
      - Funktion: **Plotten.py**. Diese Funktion wurde entwickelt, um zwei Funktionen zu implementieren, die in der Verkehrsanalyse verwendet werden können. Die erste Funktion anzahlFahrzeugeProRichtung dient dazu, die Anzahl der Fahrzeuge pro Richtung (Einfahrt und Ausfahrt) zu zählen und in einem Balkendiagramm darzustellen. Die zweite Funktion anzahlFahrzeugeProMinute zielt darauf ab, die Anzahl der erkannten Fahrzeuge pro Minute für zwei verschiedene Erkennungsmethoden (in diesem Fall OpenCV und YOLO) zu berechnen und ebenfalls in einem Balkendiagramm anzuzeigen.
      - Funktion: **data_plot.py**. Diese Funktion wurde entwickelt, um eine Echtzeitvisualisierung der Anzahl der detektierten Fahrzeuge über die Zeit mithilfe von OpenCV zu ermöglichen. Die Motivation hinter diesem Code liegt darin, einen interaktiven und visuell ansprechenden Ansatz zur Überwachung des Fahrzeugverkehrs in Echtzeit bereitzustellen. Dies kann nützlich sein für die Überwachung von Verkehrssituationen, Analyse des Fahrzeugflusses oder Evaluierung von Detektionsalgorithmen. Leider macht diese Funktion das Programm langsamer und funktioniert die Darstellung der Graphen manchmal nicht.
      - Funktion: **1_mohammad_funktion_befahrene_strecke.py**. Diese Funktion wurde entwickelt, um einen Algorithmus zur Extraktion einer befahrenen Strecke aus einem Video bereitzustellen. Die Motivation hinter diesem Code liegt darin, eine automatisierte Methode zu schaffen, um befahrene Bereiche in einem Video zu identifizieren und dadurch ein Region of Interest automatisch zu finden. 
      - <center><img src="resources/result_1_mohammad_funktion_befahrene_strecke.png" alt="resources/result_1_mohammad_funktion_befahrene_strecke.png" /> </center>
        Die Funktion hatte die Idee als Grundlage, ein Frame von einem beliebigen Frame mit festem Abstand abzuziehen, um als Resultat die unveränderten Pixels auf Null zu setzen und die Pixels, die eine Veränderung erfahren haben, in einem Array abzuspeichern. Danach werden die gespeicherten Arrays thresholder, um daraus binäre Werte zu bekommen. Danach gab es zwei Methoden die befahrenen Strecken zu extrahieren, entweder durch einfaches Addieren aller resultierenden Bilder mit anschließedem Thresholden, oder die einzelnen Pixels aller Bilder miteinander so zu vergleichen, dass die Pixels, die am meisten aufgetreten sind beizubehalten. Die erste Methode ist auf Grund nicht stabiler Aufnahme der Kreuzung gescheitert. Die zweite Methode hat nur an viel befahrenen Stellen gute Ergebnisse gezeigt, jedoch ist diese Methode extrem rechenintensiv aufgrund der Vergleich-Funktion.
      - Funktion: **2_mohammad_jan_function.py**. Diese Funktion basiert auf Grundlage der Funktion *motion_extraction*. Danach werden die resultierenden Bilder einfach in einer Schleife zu einen einzigen Array zusammen addiert. Auf dieses Array wird in jeder Iteration die morphologische Funktion 'Öffnen' angewendet. Nach einer bestimmten Anzahl an Iterationen wird das resultierende Bild thresholded. Danach werden die Ecken in diesem Bild mithilfe der Funktion 'CannyEdge' erkannt und noch mal die morphologische Funktion 'Öffnen' angewendet. Am Ende werden die wesentlichen Linien durch die Funktion 'HoughLinesP' erkannt. 
      - <center><img src="resources/result_2_mohammad_jan_function.png" alt="result_2_mohammad_jan_function" /></center>
      -  Diese Funktion hat bessere Ergebnisse als die vorige Funktion geliefert und einige Kanten der befahrenen Strecken konnten erkannt werden.



* Anschließend wurden die Ansätze mit den visuell besten Ergebnissen ausgewählt und zu dem folgenden Hauptprogramm zusammengefügt: 
  
   * Folgend sind die Ergebnisse der OPenCV Implementierung des finalen Programmes zu sehen: 
   
      <center><img src="resources/OpenCV_Ergebnis.png" alt="Bildschirmfoto 2024-03-01 um 19.52.23" /></center>
   
   * Ergebnisse von YoloV8
   
   * <center><img src="resources/Yolo22.png" alt="Bildschirmfoto 2024-03-01 um 19.52.23" /></center>
   


## 4. Evaluation

Die Funktion extract_background() funktioniert unter der Annahme, dass das Video ausreichende lang ist. Beim kurzen Videos, wo Autos lange am Ampel stehen, werden die Stehende Autos als Teil des Hintergrund erkannt.
<center><img src="resources/Background_Problem.png" alt="Einschränkung Hintergrund Erkennung" /></center>

Die Funktion find_Stats_point() bietet unter beschränkten bedingungen relative gute Ergebnisse. Die Schwächen der Funktion resultieren aus der berechneten Maske zu Identifizierung der still stehenden Autos. Die Schwächen sind:
•	Fußgänger werden manchmal als Objekte identifiziert. Das passiert hauptsächlich, wenn ein Person an der Ampel, die am nächsten zu der Kamera ist, steht. Die erkannte Contour Fläche der Person ist größer als den definierten Threshold und das Person wird von dem Programm erfasst.
<center><img src="resources/Fussgaenger_problem.PNG" alt="Einschränkung Fußgänger" /></center>
•	Staus, die nicht an der Ampel vorkommen, werden auch von der Maske erfasst. Das kann wie bei dem vorherigen Punkt zur Verschiebung des ROIs führen

•	Störungen von der Umgebung beispielweise, vom Wind (Kamerawackeln) oder von der Sonne (Änderung im Tageslicht). Die Störung verursachen Unterschiede zum Hintergrund Bild, was später von der Maske erfasst werden.
 <center><img src="resources/Einschränkung_Umgebung.PNG" alt="Einschränkung durch Tageslicht" /></center>
•	Meherere nebeneinander stehende Autos werden als ein Objekt erkannt. Das ist aber für die Berechnung des ROIs nicht relevant, da die Position der Punkten hauptsächlich für die Berechnung benötigt wird. 

•	Fahrzeuge werden manchmal als meherere Objekte erkannt werden. Der Grund dafür lässt sich allerdings nicht erklären.

•	Manche Fahrzeuge, die hinter einem Objekt in dem Hintergrund stehen, werden nicht erkannt. Da die Contour Fläche wird durch den Hintergrund Objekt verdeckt. Das führt dazu, dass die übrig gebliebene Fläche nicht größer als den Threshold ist und das Objekt nicht erkannt wird.
 <center><img src="resources/Einschraenkung_verdeckt.PNG" alt="Einschränkung durch Hintergrund Objekte" /></center>

Der hauptsächliche Grund für die Schwächen lässt sich einfach durch die verwendete Methodik erklären. Der Methodik analysiert die Bewegungen in dem Frame und es wird nicht identifiziert, ob der Bewegung aus einem Auto oder anderen Objekt im Frame sind. Um die Schwächen zu beheben bzw. verringern soll zuerst eine Methodik zur akkuraten Identifizierung den Fahrzeugen erzeugt werden, Biespielweise mithilfe von Machine Learning. Und dazu auch die Bewegungsanalyse. Das ROI kann auch direkt mit anderen Machine Learnig algorithmus bestimmt werden, in dem das Algorithmus direkt den Ampelkreuzungsberich identifiziert.

Wegen den genannten Schwächen sind die Ergebnisse von den automatischen ermittelten ROIs nicht sehr genau, weshalb wir selbst definierte ROIs verwenden. 

Das Erfassen der Fahrzeuge mithilfe von OpenCV funktioniert unter der Annahme, dass die Objekte von der Maske fehlerfrei erkannt werden. Es wurden zwei Methoden für das Erstellen der Maske verwendet. Mit beiden Masken können die Objekte erfasst werden, sodass einzelne, einfache Fahrzeuge fehlerfrei detektiert werden (siehe Abbildung A). Letztendlich wurde sich aber für Methode 2 entschieden,da bei dieser weniger Fahrzeuge falsch erkannt wurden. Unter Zuhilfenahme der Funktion *point_inside_polygon()* können Fehlerkennungen, die bei komplexeren Fahrzeugen auftreten (siehe Abbildung B) eliminiert werden. Probleme treten vor allem bei der Erkennung von Fahrzeugen, die durch andere Fahrzeuge verdeckt werden, auf (siehe Abbildung C). 
<center><img src="resources/Evaluation_Noah.png" alt="Evaluation_Noah" /></center>

Die Wegverfolgung der Fahrzeuge mit der Funktion *dist_to_line()* funktioniert ebenfalls unter der Annahme, dass die Fahrzeuge fehlerfrei erkannt werden. Dies ist in den meisten Fällen aber nur bei einzelnen, unverdeckten Fahrzeugen der Fall. Wenn dem Fahrzeug einmal eine neue ID aufgrund des Objektverlusts durch Fehler in der Maske zugewiesen wurde, wird die zweite Linienüberquerung, also die Ausfahrt, als erste Überquerung erkannt. Vermeiden lässt sich dies nur, indem das Fahrzeug über den gesamten Verlauf erkannt wird.
Dem zuvor beschrieben Objektverlust wirkt die Funktion *point_inside_polygon()* wie bereits beschrieben entgegen. Allerdings kann diese auch dazu führen, dass ein innerhalb der Kreuzung verdecktes Fahrzeug anschließend nicht mehr neu erkannt wird oder zwei erst in der Kreuzung neu erkannte Fahrzeuge dieselbe ID erhalten (siehe 2Autos_1ID). 

<center><img src="resources/2Autos_1ID.png" alt="2Autos_1ID" /></center>

Insgesamt kann die Objekterkennung mit OpenCV unter der Voraussetzung eindeutig erkannter Objekte den Verkehrsfluss der Kreuzung besser als die Objekterkennung mit YOLO

Die Implementierung von Yolo hatte auch einige Schwächen, so wurden beispielweise Objekte die sich gegenseitig verdeckt haben nicht mehr erkannt und die erkannte Instanz ging verloren und wurde danach als neue Insanz erkannt. Auch wurden anfänglich stillstehende Objekte nicht erkannt, Gründe hierfür könnten sein, dass die Encoder Schicht der Yolo Implementierung teilweise auf Bewegungen im Bild basiert. Um diese Fehler zu vermeiden hätte man ein besser trainiertes Modell benutzen können. Beispiele hierfür sind in den folgenden Bildern kenntlich gemacht. In Bild a) und b) kann man sehen , dass Objekte die sich seit Beginn des Videos nicht bewegt haben, auch nicht erkannt werden. In den Bildern c) und d) kann man erkennen sich durch die Überlappung von Objekten eine Instanz in zwei Instanzen teilt, wodurch auch die Verkehrsanalyse basierend auf der Anzahl der Instanzen in der Kreuzung fehlerhaft wird. 

<center><img src="resources/Yolo Fehler.png" alt="Yolo Fehler" /></center>


-  Getestet wurden mehrere Videos die über Youtube-Links oder als MP4 an das Programm übergeben wurden. Da die Installation der für Youtube Video notwendigen Bibliothek teilweise Probleme auftraten, haben wir uns dazu entschieden nur MP4-Videos als Material zu verwenden.
   -  <u>Beispiellinks:</u>
      -  https://www.youtube.com/watch?v=2X27I6BAJcI

-  Hiermit sollte verdeutlich werden welche Stärken und Schwächen klassische Ansätze und derzeitige Machine Learning Ansätze haben
   -  <u>Klassisch:</u>
      -  Stärken: 
         -  Wenn nicht bewegte Objekte bewegte Objekte nur teilweise verdecken, so gehen diese nicht verloren und die Bounding Box bleibt bestehen
         -  unter optimalen Bedingungen bessere Performance
         -   alle bewegten Objekte werden im Bild erkannt
         -  nachvollziehbarer Lösungsweg, keine Blackbox wie ML
      -  Schwächen: 
         -  im worst-case deutlich schlechtere Performance (min. 45fache Bearbeitungsdauer)
         -  keine Klassifizierung
         -  nur unter begrenzten Umgebungsbedingungen einsetzbar (Sonne/Schatten Wechsel)
         -  Schatten von Fahrzeugen verfälschen die Boundingbox und die Ergebnisse
   -  <u>Machine Learning:</u>
      -  Stärken: 
         -  Es werden bewegte Objekte im Bildbereich erkannt und einer Klasse wie z.B. Auto, Motorradfahrer, Truck, Laster oder Fußgänger zugeordnet. Auch werden zwei Objekte bei kleineren Überschneidungen noch differenziert. Die Modelle funktionieren "Out-of-the-Box" sehr gut und müssen für den Anfang nicht zusätzlich nachtrainiert werden. 
      -  Schwächen: 
         -  Fehler können auftreten, wenn Fahrzeuge sich zu sehr überscheiden oder von in den Bildbereich ragenden Schildern, Laternen oder Ampeln teilweise verdeckt werden, wodurch Yolo das Fahrzeug während der Verdeckung als Instanz verliert und danach zwar wiedererkennt, aber als neue Insanz, was die Werte der Verkehrsanalyse verfälscht. 
   -  Abschließender Absatz: 
      -  Hauptsächliche Unterschiede von ML und OpenCV Umsetzungen, u.a. Performance
      -  Hiefür wurden zwei Videos verglichen, eines mit einer bewölkten Umgebung und eines mit direkter Sonneneinstrahlung und resultierendem Schattenwurf
      -  OpenCV und ML könnten sich gegenseitig Ergänzen, sodass YoloV8 z.B. die Boundingbox bei teilweise Verdeckung nicht verliert und die klassifizierte Instanz erhalten bleibt
      -  ML Ansätze sind deutlich flexibler und können ohne größere Anpassungen universeller eingesetzt werden, bei OpenCV wäre eine zusätzliche Kontrolle und parametrierung notwenig

-  <u>Laufzeit:

   -  Macbook Pro M1 (links Test 1, rechts Test 2)

      <img src="user_results/graph_MBPro_Test1.png" width="425"/> <img src="user_results/graph_MBPro_Test2.png" width="425"/> 

   -  Ryzen 3600x with RTX4070 (links Test 1, rechts Test 2)

      <img src="user_results/graph_RTX4070_Test1.png" width="425"/> <img src="user_results/graph_RTX4070_Test2.png" width="425"/> 

   -  Weitere Ergebnisse unserer Arbeit sind in dem Ordner user_results zu finden. Hierbei kann jeder Nutzer sich auch selber Ergebnisse generieren um die Performance auf dem zu untersuchenden Endgerät zu testen

* <u>Was hätte wie besser sein können?</u>: 
   * Differenzierung von bewegten Objekten durch ergänzung des mit Lukas Kanade Dense Optical Flow, um über Skalarfelder eventuell die Objekte zu differenzieren
   * Rechenzeit und Stabilität der OpenCV Implementierung verbessern 
   * die OpenCV Implementierung für den Einsatz bei Nacht ergänzen
   * erkennung von Fahrzeugen nicht nur duch Bewegungsanalyse sondern auch durch Featureerkennung um die zuverlässigkeit der OpenCV umsetzung zu verbessern
* Wer könnte von Ihrem Projekt profitieren? 
   * Verkehrsplaner und Verkehrsüberwachungssysteme
   * Unfallerkennugnssysteme, wenn der Algorithmus zuverlässiger wäre, könnte festgestellt werden wenn ein Auto zu lange (Unfall) auf der Kreuzung steht

## 5. Programmdokumentation
Dokumentieren Sie Ihren Quellcode leserlich und laden Sie alle relevanten Dateien und Datensätze in den "project"-Ordner hoch (sollte ihr Projekt mehr als 100MB verbrauchen, so legen Sie dieses bitte extern ab (bspw. GitHub, GDrive oder ähnliches) und verlinken dieses im "project"-Ordner.


# Arbeitsmatrix
Listen Sie hier alle relevanten Features als Liste mit sinnvoller Reihenfolge auf mit Angabe wer das Feature implementiert hat und zu welchem Anteil. Die Features sollen aus der Beschreibung des Projektes oben klar hervorgehen und nur die größeren Arbeitspakete umfassen (ca. 4-8 Features).
Bei starker Ungleicheit der Features behalten wir uns eine individuelle Notenvergabe vor.

Bspw.

- Systemarchitektur [MAIN] (Noah 32,5%, Mohammad 10%, Mostafa 25%, Jan 32,5%)
- Objekterkennung mit OpenCV (Noah 100 %)
- Fahrzeugweg tracken mit OpenCV (Noah 100 %)
- Backround Separation mit OpenCV Funktion(25% Jeder)
- Backround Seperation mit auf anderen Wegen [eigene Implementierung] (Mostafa 100%)
- automatische ROI über stillstehende Fahrzeuge (Mostafe 100%)
- YOLOv8 und Tests  (Jan 100%)
- Lukas Kanade Funktion und Tests (Jan 100%)
- Erkennung von befahrenen Strecken (Mohammad 100%)
- Statistische Auswertung (Mohammad 100%)

# Installation

```
git clone https://github.com/Mohammadalbittar/DBV_WS23-24.git
```

```
pip install -r requirements.txt
```

## Terminal Start Syntax

```
python main.py "file title" "path_to_file" [True/False -> Save Video File] [True/False -> Live Output] [FPS != INT i.e. 30]
```



## For Video and System Performance Output
Test the first 10 seconds of a demo video
```
python main.py "demo_title" 'resources/video2.mp4' True False 10
```

## For live Viewing with not Output
Just live view the result with no Saved File. Press Escape to abort
```
python main.py "" 'resources/video2.mp4' False True 0
```

# Eklärungsvideo

```
https://drive.google.com/drive/folders/1oj64TH0qNlh3y8DgVZZC_jRUDCNw7ZyY?usp=sharing
```

