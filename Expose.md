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



# Expose: Analyse einer Kreuzung - Vergleich von Machine Learning zu Nicht-Machine-Learning Methoden

**Einleitung:** Im Rahmen dieses Projekts liegt der Fokus auf der Analyse von Kreuzungen, einem zentralen
Aspekt moderner Verkehrsinfrastrukturen. Die Problematik umfasst verschiedene Herausforderungen,
darunter die Hintergrunderkennung, die Detektion nicht bewegter Fahrzeuge, die Störungen durch bewegte
Fremdobjekte sowie die präzise Erfassung von Straßen und Straßenmarkierungen.

**Problemstellungen:**

1. _Hintergrunderkennung als Grundlage:_ Die grundlegende Herausforderung besteht in der präzisen
    Identifikation des Hintergrunds, um eine zuverlässige Differenzierung zwischen statischen und
    bewegten Objekten zu ermöglichen.
2. _Detektion nicht bewegter Fahrzeuge:_ Die Erkennung von unbewegten Fahrzeugen stellt eine
    anspruchsvolle Aufgabe dar, insbesondere wenn diese aufgrund ihrer Umgebung schwer zu
    differenzieren sind.
3. _Störungen durch bewegte Fremdobjekte:_ Die Anwesenheit von bewegten Fremdobjekten wie Fußgängern
    und Radfahrern erzeugt Störungen, die die Effizienz der Verkehrsflussanalyse beeinträchtigen können.
4. _Erfassung von Straßen und Straßenmarkierungen:_ Eine präzise Identifikation von Straßenkonturen und
    Markierungen ist entscheidend für eine umfassende Analyse der Verkehrssituation.

**Ziele/Output des Projekts:** Das übergeordnete Ziel dieses Projekts besteht in der Entwicklung und
Evaluation von Methoden, die sowohl auf Machine Learning- als auch auf Nicht-Machine-Learning-Ansätzen
basieren. Die angestrebten Ergebnisse sind:

1. _Fahrzeuge je Minute erkennen:_ Implementierung von Algorithmen zur präzisen Erkennung von
    Fahrzeugen, unabhängig von deren Bewegungsstatus, um eine quantitative Analyse des
    Verkehrsaufkommens zu ermöglichen.
2. _Verkehrsfluss über Heatmap visualisieren:_ Entwicklung einer Heatmap zur visuellen Darstellung des
    Verkehrsflusses an der Kreuzung, um Muster und Intensitäten zu identifizieren.
3. _Mittlere Geschwindigkeit:_ Berechnung der mittleren Geschwindigkeit der erkannten Fahrzeuge, um
    Einblicke in die Dynamik des Verkehrsflusses zu gewinnen.

**Methodischer Ansatz:** Der methodische Ansatz integriert eine Vielfalt von Techniken, sowohl auf Basis von
Machine Learning als auch von traditionellen Verfahren, um eine umfassende Lösung für die genannten
Problematiken zu erarbeiten.

**Nutzen und Anwendung:** Die Ergebnisse dieses Projekts haben das Potenzial, die Effizienz der
Verkehrsflussüberwachung an Kreuzungen zu verbessern. Die entwickelten Methoden können in
intelligenten Verkehrssystemen implementiert werden, um die Sicherheit und Leistungsfähigkeit des
städtischen Verkehrsflusses zu optimieren.

