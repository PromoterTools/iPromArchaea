# 🧬 iProm-Archaea: CNN-based Archaeal Promoter Prediction Tool

<span style="color:gray;">A deep learning webserver for identifying archaeal promoter sequences with high accuracy.</span>

---

## 🔍 Overview

**iProm-Archaea** is a CNN-based tool designed to detect **promoter sequences in archaea**, using optimal 6-mer encoding. It outperforms existing methods in accuracy, precision, and AUC, and is freely available via [webserver](https://bec7-210-39-1-111.ngrok-free.app/) and GitHub.

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

