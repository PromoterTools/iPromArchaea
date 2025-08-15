# 🧬 Characterization of Archaeal Promoters Using explainable and web-based CNN model

<span style="color:gray;">A deep learning webserver for identifying archaeal promoter sequences with high accuracy.</span>

---
## 🖥️ Updated
We have updated our webserver link https://1289db90d996.ngrok-free.app/
## 🔍 Overview

**iProm-Archaea** is a CNN-based tool designed to detect **promoter sequences in archaea**, using optimal 6-mer encoding. It outperforms existing methods in accuracy, precision, and AUC, and is freely available via [webserver](https://1289db90d996.ngrok-free.app/) and GitHub.

We also annotate **~586,455 promoter sequences** from **478 archaeal genomes**, contributing a significant resource for the community.

---

## 🚀 Key Features

- 📊 **State-of-the-art CNN** model with 91.69% accuracy (training) and 88.95% (independent test)
- 🔢 **6-mer encoding** found to outperform DDS, one-hot, and other schemes
- 🌍 **Cross-organism analysis** shows promoter diversity across domains
- 🧪 **Evaluation metrics**: Accuracy, Precision, Recall, Specificity, AUC
- 🖥️ **User-friendly webserver** with FASTA input support and DDS-based plots
- 📂 Annotated **586K+ archaeal promoters** from 478 species (available on GitHub)

---

## 🧰 Technologies Used

<span style="color:green;">Python, TensorFlow/Keras, Scikit-learn</span>  
<span style="color:blue;">Flask</span> for web deployment  
<span style="color:orange;">BioPython</span> for sequence parsing  

---

## 📁 Repository Contents
📄 app.py                       → Flask webserver script for running the iProm-Archaea tool
📄 requirements.txt            → Python dependencies for the project
📄 .gitignore                  → Git ignored files configuration

📁 templates/                  → HTML templates for the web interface

📄 model_cnn.weights.h5        → Trained CNN model weights

📄 Promoters_training.txt      → Positive training set (promoter sequences)
📄 negative_training.txt       → Negative training set (non-promoter sequences)

📄 Arabidopsis thaliana fasta.txt     → Eukaryotic promoter dataset (test)
📄 h.Spanies EPDnew.txt               → Human promoter dataset (test)
📄 Macaca mulatta (rhesus maca ...    → Primate promoter dataset (test)

📦 Predicted Promoters.zip     → Annotated archaeal promoter predictions (586,455 sequences)

📄 README.md                   → This file (project overview, usage, license, etc.)
📜 License: Original data: CC-BY-NC-ND (non-commercial, no derivatives). Our code/processed data: MIT License (no restrictions)."

