Exposé Kreuzung Verkehrsanalyse
 
In unserem Forschungsprojekt konzentrieren wir uns auf die Anwendung herkömmlicher Computer Vision-Methoden zur Verkehrsanalyse, ohne auf fortgeschrittene Techniken wie Machine Learning oder Künstliche Intelligenz zurückzugreifen. Durch die gezielte Nutzung von Bildverarbeitungsalgorithmen und klassischer Mustererkennung streben wir an, präzise und effiziente Lösungen für die Verkehrsüberwachung und mögliche Verkehrsoptimierung zu entwickeln. Unser Ansatz betont die Robustheit und Rechenleistung konventioneller CV-Methoden, um eine kosteneffektive und zugängliche Alternative zu modernen, ressourcenintensiven Ansätzen zu bieten.



Konkretisieren : 
- Trajektorie + Geschwindikeit 
- Verkehrsfluss
- Erkennen von Autos/Personen/LKW/ Roller (Machine Learning)
- Erkennen von bewegten Objekten ohne Machine Learning

Schritte und Probelem: 
- Objekte im Bild finden, Straßenbegrenzung erkennen, (Fußweg erkennen)
- Fußgänger/ Autos unterscheiden (ohne ML schwer, mit ML easy)
- Statistik Verkehrsfluss 
- Geschwindigkeit bestimmen -> (Dense Optical FLow (Lucas-Kanade method))


Hallo Jan,

klingt soweit gut. Versucht das Ganze vielleicht bis zur Exposé Abgabe noch ein wenig zu konkretisieren,
falls möglich. Vielleicht könnt ihr schon verschiedene Schritte und Probleme identifizieren die ihr auf
jeden Fall lösen müsstet, und was eventuell optional wäre.
Für das initiale Detektieren von Objekten würde ich fertige ML Modelle (z.B. YOLO)
vielleicht nicht komplett rauswerfen, die machen einem das Leben schon deutlich einfacher.
Es wäre auch super, wenn ihr nochmal kurz beschreiben könntet, was ihr genau als
Output anstrebt. Trajektorien von Fahrzeugen? Zustände (wartend, über rote Ampel gefahren...)?
Statistiken (z.B. durchschnittlicher "Fluss" in und aus der Kreuzung)?

Viele Grüße
Fabian
