# Evaluation-VLMs
## Bachelorarbeit

Title: __Evaluation bestehender Modelle zur Gestenerkennung auf den Videodaten des Va.Si.Li-Lab Datensatzes__

__Johann Wolfgang von Goethe Universität Frankfurt am Main__  
Name: Jaehyun Shin  
MatrikelNr.: 6555216  
Studiengang: B.Sc. Informatik

Text Technology Lab  
__Betreuer: Prof. Dr. Alexander Mehler__  

---

**Hinweis**: Dies Repository enthält keinen Datasatz, der für die Bachelorarbeit genutzt wurde, aufgrund der Lizenz.

### Kurze Beschreibung über den Datensatz:
Der Datensatz besteht auf Videos und .eaf Datei.
Eine .eaf Datei enthält Labels für ein Video.

__naming:__
- Video: angle_0_final.mp4
- {annotator}_angle_0_final.eaf

Die Labels sind wie folgt definiert:
- Representing
- Drawing
- Indexing
- Molding
- Emblematic
- Beat
- Acting
- other

Das Label 'NoGesture' wird von mir bearbeitet und hinzugefügt für das Experiment.
Wenn keine 'NoGesture' Labels entstehen, verursacht sehr großes Ungleichgewichtsproblem.
  
### **Experiment Schritte:**
1. 0.extrakt.py: erstellen Segments und ground-truth file (Index und label) für Envision und VLMs.
.eaf Datei enthält hauptsächlich 'error' label aber in diesem Phase wird die 'error' Labels entfernt,
weil die 'error' Labels in diesem Experiment nicht berücksichtigt werden. Ausgabe: **clips.csv** und **output(ordner)/{index}.mp4**
2. 6.sample.ipynb: (jupyter notebook file): Führe EnvisionHGDetector aus
-> erstellen **{index}.mp4_predictions.csv** und -segment.csv. Aber für das Experiment wird nur die predictions.csv benutzt.
3. makebinarygt.py: Wandele die Label von clips.csv(ground-truth) aus Schritt 1. zu 'Gesture' oder 'NoGesture'. Ausgabe: **binarygt.csv**
4. extract.py: Extrahiere nur Index und Label von clips.csv, binarygt.csv aus Schritt 1, 3.
Ausgabe: **clipsextract.csv, binarygtextract.csv** clipsextract.csv ist für 9class gt und binarygtextract.csv ist für binary.
(Hinweise: Bei der jeden Nutzung dieses Codes muss man den Eingabename von Datei "csv_files = sorted(cwd.glob(**"truelabel.csv"**))" selbst eingeben.
5. make_envision_label.py: Erstelle Vorhersagen von EnvisionHGDetector aus dem Schritt 2. ({index}.mp4_predictions.csv). Ausgabe: **truelabel.csv oder envisionpred.csv**
+ Mit Nutzung von extract.py extrahiere Index und Label aus **truelabel.csv oder envisionpred.csv**.
6. pure8class.py: Extrahiere Videosegmente und Labels, die kein NoGesture sind. Ausgabe: **angle_0_pure/{index}.mp4, 8class.csv**
+ Mit Nutzung von extract.py extrahiere Index und Label aus **8class.csv**. Die in diesem Schritt erstellte Segmente und Labels sind für Experiment für 8class segementieren.
7. In der **eschar** Umgebung befinden sich die Ground-truth Datensätze in Ordner **Jayproject/qwen2.5/models/testdata/labels** und videos
  **8class**: edab_8class(video44), video41pure, video42,43 oder edab,video41,42,43.csv (Hinweis: edab und video41.csv enthält andere Informationen außer Index und Label)
  **9class**: {video}_new8class.py
  **binary**: {video}_newbinary.py
8. __qwen2.5-VL-7B__ - qwen2.5/models/multifolder9class.py: Führe Vorhersagen und speichern in ergebnisse. (**Hinweise: "videos_to_process" sollte selbst eingeben** Liste sind im Schritt 7 geschrieben. Ausgabe: ergebnisse/{video}_new8class.csv(9class), {video}_newbinary.csv(binary), ergebnisse/purs8class/{video}.csv(8class)
9. __LLaVA-NeXT-Video__ - LLaVA-NeXT-Video/multifolder8class.py: Vorhersagen ausführen und speichern in output. Der Name der gespeicherten Daten sind ähnlich wie qwen2.5
10. __LLaVA-OneVision__ - LLaVA-OneVision/multifolder8class.py: Gleich wie LLaVA-NeXT-Video

### Evaluation:
__Für EnvisionHGDetector__
- evalenvision.py in der lokalen Umgebung. Ausgabe: __Desktop/SSH/envisionresult__ **envision_binary_summary.csv**: Accuracy, weighted-, macro-Metriken sowie Precision, Recall und F1-Score. | **binary_confusion_matrix**: Confusion Matrix

__Für VLMs__
- eval, dist in der **eschar** Umgebung Jayproject.
- Für Evaluation werden die folgende Coden benutzt - eval8class.py(zusätzlich angle), eval9class.py(zusätzlich angle), evalbinary.py, evalangle.py. Ausgabe: Evaluation (Metriken) und Confusion Matrix
- distribution graph: dist8classtotal.py, dist9classtotal.py
