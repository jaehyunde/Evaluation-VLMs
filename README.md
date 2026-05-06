# Vision-Language Model Evaluation for Gesture Recognition

Bachelor Thesis Project on Vision-Language Model Evaluation for Gesture Recognition using the Va.Si.Li-Lab Dataset.

## 📖 Thesis Title

> **Evaluation bestehender Modelle zur Gestenerkennung auf den Videodaten des Va.Si.Li-Lab Datensatzes**

## 📌 Overview

This project evaluates existing Vision-Language Models (VLMs) for gesture recognition tasks using video data from the Va.Si.Li-Lab dataset.

The goal is to analyze the performance, robustness, and applicability of multimodal models such as LLaVA and Qwen in real-world gesture understanding scenarios.

## 🧠 Tech Stack

### Programming Language
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

### AI / Deep Learning
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![LLaVA](https://img.shields.io/badge/LLaVA-6A5ACD?style=flat-square)
![Qwen](https://img.shields.io/badge/Qwen-FF6F00?style=flat-square)

### Environment
![Anaconda](https://img.shields.io/badge/Anaconda-44A833?style=flat-square&logo=anaconda&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=ubuntu&logoColor=white)

### Version Control
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)

### Data Format
![JSON](https://img.shields.io/badge/JSON-000000?style=flat-square&logo=json&logoColor=white)

### Hardware
![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white)

---

Author: Jaehyun Shin  
B.Sc. Informatik  
*Johann Wolfgang von Goethe Universität Frankfurt am Main*

Text Technology Lab  
Supervisor: Prof. Dr. Alexander Mehler

## 🎓 Thesis Status

Bachelor thesis successfully completed in April 2026.

---

**Hinweis**: Dieses Repository enthält aufgrund der Lizenz nicht den Datensatz, der für die Bachelorarbeit genutzt wurde.

### Beschreibung des Datensatzes

Der Datensatz besteht aus Videos und `.eaf`-Dateien.  
Eine `.eaf`-Datei enthält die Labels für ein Video.

__Naming:__

- Video: `angle_0_final.mp4`
- Annotation: `{annotator}_angle_0_final.eaf`

Die Labels sind wie folgt definiert:

- Representing
- Drawing
- Indexing
- Molding
- Emblematic
- Beat
- Acting
- other

Das Label `NoGesture` wurde von mir für das Experiment bearbeitet und hinzugefügt.  
Wenn keine `NoGesture`-Labels entstehen, verursacht dies ein sehr großes Ungleichgewichtsproblem.

### **Experiment-Schritte**

1. `0.extrakt.py`: Erstellt Segmente und eine Ground-Truth-Datei mit Index und Label für Envision und die VLMs.  
   Die `.eaf`-Datei enthält hauptsächlich das Label `error`. In dieser Phase werden die `error`-Labels jedoch entfernt, da sie in diesem Experiment nicht berücksichtigt werden.  
   Ausgabe: **`clips.csv`** und **`output/` (Ordner) / `{index}.mp4`**

2. `6.sample.ipynb` (Jupyter-Notebook-Datei): Führt den EnvisionHGDetector aus.  
   Erstellt **`{index}.mp4_predictions.csv`** und **`{index}.mp4_segment.csv`**. Für das Experiment wird jedoch nur die `predictions.csv` benutzt.

3. `makebinarygt.py`: Wandelt die Labels aus `clips.csv` (Ground Truth) aus Schritt 1 in `Gesture` oder `NoGesture` um.  
   Ausgabe: **`binarygt.csv`**

4. `extract.py`: Extrahiert nur Index und Label aus `clips.csv` und `binarygt.csv` aus Schritt 1 und 3.  
   Ausgabe: **`clipsextract.csv`**, **`binarygtextract.csv`**  
   `clipsextract.csv` ist für die 9-Class-Ground-Truth und `binarygtextract.csv` für die binäre Ground-Truth.

   **Hinweis**: Bei jeder Nutzung dieses Codes muss der Eingabename der Datei selbst eingegeben werden, z. B.:  
   `csv_files = sorted(cwd.glob("truelabel.csv"))`

5. `make_envision_label.py`: Erstellt Vorhersagen des EnvisionHGDetector aus Schritt 2 (`{index}.mp4_predictions.csv`).  
   Ausgabe: **`truelabel.csv`** oder **`envisionpred.csv`**

   Zusätzlich wird mit `extract.py` Index und Label aus **`truelabel.csv`** oder **`envisionpred.csv`** extrahiert.

6. `pure8class.py`: Extrahiert Videosegmente und Labels, die kein `NoGesture` sind.  
   Ausgabe: **`angle_0_pure/{index}.mp4`**, **`8class.csv`**

   Zusätzlich wird mit `extract.py` Index und Label aus **`8class.csv`** extrahiert.  
   Die in diesem Schritt erstellten Segmente und Labels sind für das 8-Class-Experiment segmentiert.

7. In der **`eschar`**-Umgebung befinden sich die Ground-Truth-Datensätze im Ordner **`Jayproject/qwen2.5/models/testdata/labels`** und die Videos.

   **8class**: `edab_8class` (`video44`), `video41pure`, `video42`, `video43` oder `edab`, `video41`, `video42`, `video43.csv`  
   **Hinweis**: `edab` und `video41.csv` enthalten außer Index und Label auch andere Informationen.

   **9class**: `{video}_new8class.py`  
   **binary**: `{video}_newbinary.py`

8. __Qwen2.5-VL-7B__ – `qwen2.5/models/multifolder9class.py`: Führt Vorhersagen aus und speichert sie in `ergebnisse`.  
   **Hinweis**: `videos_to_process` muss selbst eingegeben werden. Die Liste ist in Schritt 7 beschrieben.  
   Ausgabe: `ergebnisse/{video}_new8class.csv` (9class), `{video}_newbinary.csv` (binary), `ergebnisse/purs8class/{video}.csv` (8class)

9. __LLaVA-NeXT-Video__ – `LLaVA-NeXT-Video/multifolder8class.py`: Führt Vorhersagen aus und speichert sie in `output`.  
   Die Namen der gespeicherten Dateien sind ähnlich wie bei Qwen2.5.

10. __LLaVA-OneVision__ – `LLaVA-OneVision/multifolder8class.py`: Gleich wie LLaVA-NeXT-Video.

### Evaluation

__Für EnvisionHGDetector__

- `evalenvision.py` in der lokalen Umgebung.  
  Ausgabe: **`Desktop/SSH/envisionresult`**

  - **`envision_binary_summary.csv`**: Accuracy, weighted- und macro-Metriken sowie Precision, Recall und F1-Score
  - **`binary_confusion_matrix`**: Confusion Matrix

__Für VLMs__

- `eval` und `dist` in der **`eschar`**-Umgebung unter `Jayproject`.
- Für die Evaluation werden die folgenden Codes benutzt:  
  `eval8class.py` (zusätzlich Angle), `eval9class.py` (zusätzlich Angle), `evalbinary.py`, `evalangle.py`  
  Ausgabe: Evaluationsergebnisse (Metriken) und Confusion Matrix
- Distribution Graphs: `dist8classtotal.py`, `dist9classtotal.py`
