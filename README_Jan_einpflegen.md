## Formulierung des Vorgehens

Vergleich der Verschiedenen Methoden für die Hintergrundsegmentierung. Hier wurden die Verschiedenen Umsetzungen von dem OpenCV internen BackgroundSubtractor getestet, MOG2, KNN, CNT und GMG.



Die Verschiedenen Umsetzungen lieferten gemischte Ergebnisse, da gelegentlich der Output flackerte oder wenn Autos über lägere Zeit still standen diese als Hintergrund klassifiziert wurden und somit aus der Ausgabe verschwanden. Dies hat den Hintergrund, dass die Umsetzungen des BackgroundSubtractor mit einer History arbeiten und hierbei jedem Pixel einen, auf Wahrscheinlichkeit über die Bildhistorie betrachtet, basierenden Wert zuweisen. ![BackroundSub](/Users/jansbiegay/PycharmProjects/DBV_WS23-24/assets/BackroundSub.png)



Als nächstes wurde eine vergleichsweise simple Methode für die Motion Extraction aus Video getestet. Diese Methode basiert auf der Idee der Quelle [Quelle]. Hierbei werden zwei aufeinander folgende Frames miteinander verrechnet. Bewegt sich ein Objekt im Bild, entsteht eine Kante die um die Pixelwertänderung vom Mittelwert des Maximalwertes des Datentyps (Beispiel: Datentyp uint8 -> Mittelwert 128) abweicht. Subtrahiert man nun den Mittelwert vom Bild und bildet den Absoluter. so erhält man ein schwarzes Bild wo nur durch Bewegung erzeugte Kanten vom 0 Wert abweichen. Durch das zusätzliche Anwenden des OpenCV Threshholding Algorithmus auf das Ergebnis, kann Rauschen und minimale Änderungen rausgefiltert werden und man erhält eine binäre Masse mit "Bewegung" =1 und "Unbewegt" = 0.

![Motion](/Users/jansbiegay/PycharmProjects/DBV_WS23-24/assets/Bildschirmfoto 2024-02-29 um 20.06.35.png)



Anschließend wurde noch eine Umsetzung des Lukas Kanade Algorithmus für Dense Optical Flow von OpenCV getestet. Ziel war es, anhand eines Vektorfeldes einzelne bewegte Objekte im Bild zu segmentieren, und sich überlagernde Objekte anhand des Algorithmus zu differenzieren. Die Umsetzung hiervon war sehr rechenintensiv und langsam, weshalb für eine spätere eventuelle, alternative Verwendung unter anderem nur die Magnitude und der Winkel als Rückgabeparameter implementiert wurden. ![Lukas_Kanade](/Users/jansbiegay/PycharmProjects/DBV_WS23-24/assets/Lukas_Kanade.png)



Um die unterschiedlichen Binärmasken aus den Ergebnissen der vorhergehenden FUnktionen zu verbessern, wurden unterschieliche Clustering Methoden betrachtet. Diese waren aber aufgrund einer sehr hohen Laufzeit nicht sinnvoll anwendbar oder brauchten zur initialisierung eine Anzahl der gewünschten Cluster, was in diesem Anwendungsfall nicht sinnvoll ist. 



Für die Umsetzung mit Machine Learning wurde YoloV8 ausgewählt, da dieses vergleichsweise sehr gut Dokumentiert war und die gegebenen Modelle ausreichend trainiert waren, sodass diese auf Anhieb funktionieren konnten. Hierfür wurden auf der Grundlage von zwei Online Beispielen zwei Umsetzungen getestet, explizit wurden die Modi Predict und Track getestet, jedoch waren für unseren Anwendungsfall hier keine gravierenden Unterschiede ersichtlich.![yolo_region](/Users/jansbiegay/PycharmProjects/DBV_WS23-24/assets/yolo_region.png)





## Evaluation



Das erfassen von Fahrzeugen klappt relativ zuverlässig mit YoloV8. 

Stärken: 

Es werden bewegte Objekte im Bildbereich erkannt und einer Klasse wie z.B. Auto, Motorradfahrer, Truck, Laster oder Fußgänger zugeordnet. Auch werden zwei Objekte bei kleineren Überschneidungen noch differenziert. Die Modelle funktionieren "Out-of-the-Box" sehr gut und müssen für den Anfang nicht zusätzlich nachtrainiert werden. 



Schwächen: 

Fehler können auftreten, wenn Fahrzeuge sich zu sehr überscheiden oder von in den Bildbereich ragenden Schildern, Laternen oder Ampeln teilweise verdeckt werden, wodurch Yolo das Fahrzeug während der Verdeckung als Instanz verliert und danach zwar wiedererkennt, aber als neue Insanz, was die Werte der Verkehrsanalyse verfälscht. 



Wie könnte man den Ansatz erweitern: 

Man könnte versuchen 
