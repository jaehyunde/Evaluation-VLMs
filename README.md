# Vision-Language Model Evaluation for Gesture Recognition

Bachelor thesis project on the evaluation of existing Vision-Language Models for gesture recognition using video data from the Va.Si.Li-Lab dataset.

## 📖 Thesis Title

> **Evaluation bestehender Modelle zur Gestenerkennung auf den Videodaten des Va.Si.Li-Lab Datensatzes**

## 📌 Overview

This project evaluates existing Vision-Language Models (VLMs) for gesture recognition tasks using segmented video data from the Va.Si.Li-Lab dataset.

The goal is to analyze the performance, robustness, and applicability of multimodal models such as LLaVA and Qwen for gesture recognition in segmented video clips.

## 🤖 Evaluated Models

The following models were evaluated in this project:

- Qwen2.5-VL-7B
- LLaVA-NeXT-Video
- LLaVA-OneVision
- EnvisionHGDetector

## 🎯 Evaluation Tasks

The evaluation was conducted on the following task settings:

- **Binary classification**: `Gesture` vs. `NoGesture`
- **9-class classification**: eight gesture classes + `NoGesture`
- **8-class classification**: gesture classes only, excluding `NoGesture`

## 📊 Evaluation Metrics

The models were evaluated using the following metrics and analysis methods:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-averaged metrics
- Weighted metrics
- Confusion Matrix
- Prediction Distribution Graphs

## 🧠 Tech Stack

### Programming Language

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

### AI / Deep Learning

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
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
![CSV](https://img.shields.io/badge/CSV-217346?style=flat-square)
![ELAN](https://img.shields.io/badge/ELAN-.eaf-blue?style=flat-square)

### Hardware

![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white)

---

## 🎓 Thesis Information

**Author**: Jaehyun Shin  
**Degree Program**: B.Sc. Informatik  
**University**: Johann Wolfgang Goethe-Universität Frankfurt am Main  
**Lab**: Text Technology Lab  

**Supervisors**: Prof. Dr. Alexander Mehler, Dr. Andy Lücking  

## 🎓 Thesis Status

Bachelor thesis successfully completed in April 2026.

---

## ⚠️ Dataset Notice

**Hinweis**: Dieses Repository enthält aufgrund der Lizenz nicht den Datensatz, der für die Bachelorarbeit genutzt wurde.

## Beschreibung des Datensatzes

Der Datensatz besteht aus Videos und `.eaf`-Dateien.  
Eine `.eaf`-Datei enthält die Labels für ein Video.

### Naming

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
Ohne zusätzlich erzeugte `NoGesture`-Labels würde ein starkes Ungleichgewicht zwischen Gesture- und NoGesture-Beispielen entstehen.

---

## Experiment-Schritte

### 1. Segmentierung und Ground-Truth-Erstellung

`0.extrakt.py`: Erstellt Segmente und eine Ground-Truth-Datei mit Index und Label für Envision und die VLMs.

Die `.eaf`-Datei enthält hauptsächlich das Label `error`. In dieser Phase werden die `error`-Labels jedoch entfernt, da sie in diesem Experiment nicht berücksichtigt werden.

**Ausgabe**:

- `clips.csv`
- `output/{index}.mp4`

### 2. EnvisionHGDetector ausführen

`6.sample.ipynb` (Jupyter-Notebook-Datei): Führt den EnvisionHGDetector aus.

**Ausgabe**:

- `{index}.mp4_predictions.csv`
- `{index}.mp4_segment.csv`

Für das Experiment wird jedoch nur die `predictions.csv` benutzt.

### 3. Binary Ground Truth erstellen

`makebinarygt.py`: Wandelt die Labels aus `clips.csv` (Ground Truth) aus Schritt 1 in `Gesture` oder `NoGesture` um.

**Ausgabe**:

- `binarygt.csv`

### 4. Index und Label extrahieren

`extract.py`: Extrahiert nur Index und Label aus `clips.csv` und `binarygt.csv` aus Schritt 1 und 3.

**Ausgabe**:

- `clipsextract.csv`
- `binarygtextract.csv`

`clipsextract.csv` ist für die 9-Class-Ground-Truth und `binarygtextract.csv` für die binäre Ground-Truth.

**Hinweis**: Bei jeder Nutzung dieses Codes muss der Eingabename der Datei selbst eingegeben werden, z. B.:

```python
csv_files = sorted(cwd.glob("truelabel.csv"))

### 5. EnvisionHGDetector-Labels erstellen

`make_envision_label.py`: Erstellt Vorhersagen des EnvisionHGDetector aus Schritt 2 (`{index}.mp4_predictions.csv`).

**Ausgabe**:

- `truelabel.csv`
- oder `envisionpred.csv`

Zusätzlich wird mit `extract.py` Index und Label aus `truelabel.csv` oder `envisionpred.csv` extrahiert.

### 6. 8-Class-Datensatz erstellen

`pure8class.py`: Extrahiert Videosegmente und Labels, die kein `NoGesture` sind.

**Ausgabe**:

- `angle_0_pure/{index}.mp4`
- `8class.csv`

Zusätzlich wird mit `extract.py` Index und Label aus `8class.csv` extrahiert.

Die in diesem Schritt erstellten Segmente und Labels sind für das 8-Class-Experiment segmentiert.

### 7. Ground-Truth-Datensätze in der `eschar`-Umgebung

In der `eschar`-Umgebung befinden sich die Ground-Truth-Datensätze im Ordner:

```text
Jayproject/qwen2.5/models/testdata/labels
```

Die Videos befinden sich ebenfalls in der entsprechenden Projektumgebung.

#### 8class

```text
edab_8class (video44)
video41pure
video42
video43
```

oder:

```text
edab
video41
video42
video43.csv
```

**Hinweis**: `edab` und `video41.csv` enthalten außer Index und Label auch andere Informationen.

#### 9class

```text
{video}_new8class.py
```

#### Binary

```text
{video}_newbinary.py
```

### 8. Qwen2.5-VL-7B

`qwen2.5/models/multifolder9class.py`: Führt Vorhersagen aus und speichert sie in `ergebnisse`.

**Hinweis**: `videos_to_process` muss selbst eingegeben werden. Die Liste ist in Schritt 7 beschrieben.

**Ausgabe**:

- `ergebnisse/{video}_new8class.csv` (9class)
- `{video}_newbinary.csv` (binary)
- `ergebnisse/pure8class/{video}.csv` (8class)

### 9. LLaVA-NeXT-Video

`LLaVA-NeXT-Video/multifolder8class.py`: Führt Vorhersagen aus und speichert sie in `output`.

Die Namen der gespeicherten Dateien sind ähnlich wie bei Qwen2.5.

### 10. LLaVA-OneVision

`LLaVA-OneVision/multifolder8class.py`: Gleich wie LLaVA-NeXT-Video.

---

## Evaluation

### Für EnvisionHGDetector

`evalenvision.py` in der lokalen Umgebung.

**Ausgabeordner**:

```text
Desktop/SSH/envisionresult
```

**Ausgaben**:

- `envision_binary_summary.csv`: Accuracy, weighted- und macro-Metriken sowie Precision, Recall und F1-Score
- `binary_confusion_matrix`: Confusion Matrix

### Für VLMs

`eval` und `dist` in der `eschar`-Umgebung unter `Jayproject`.

Für die Evaluation werden die folgenden Codes benutzt:

- `eval8class.py`  
  zusätzlich Angle
- `eval9class.py`  
  zusätzlich Angle
- `evalbinary.py`
- `evalangle.py`

**Ausgabe**:

- Evaluationsergebnisse, also Metriken
- Confusion Matrix

### Distribution Graphs

Für die Prediction Distribution Graphs werden die folgenden Codes benutzt:

- `dist8classtotal.py`
- `dist9classtotal.py`
