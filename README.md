# 🔍 Loss Landscape Geometry & Optimization Dynamics  
### *A Practical Framework Connecting Curvature, SGD Noise, Generalization, and Architecture Design*

This repository implements a rigorous framework for analyzing neural network **loss landscapes**, including curvature (Hessian spectrum), sharpness, and 2D loss geometry, and relates these properties to SGD optimization dynamics and generalization. Experiments are performed on MNIST across two architectures (MLP vs CNN) and two optimization regimes (high-noise vs low-noise SGD).

The framework is modular, reproducible, and easily extendable to **OCR models**, **LLMs**, **Transformers**, and large-scale deep networks.

---

## 🚀 Highlights

- ⚡ Efficient **Hessian–vector product** curvature estimation  
- 📈 **Power iteration** to compute largest eigenvalue (λ_max)  
- 🔎 **Random-direction sharpness** computation  
- 🌄 **2D loss landscape slicing & visualization**  
- 🧱 Architecture comparison: **MLP vs CNN**  
- 🔥 High-noise vs low-noise SGD generalization study  
- 📊 Clean PyTorch experiments with CSV logging  
- 📄 Complete LaTeX research report  

---

## 📁 Repository Structure

├── run_experiments.py # Train models, log loss/accuracy/λ_max/sharpness
├── compare_landscape_results.py # Generate comparison plots (loss, accuracy, λ_max, sharpness, gap)
├── README.md # This file
├── logs_experiments/ # Auto-created logs for each experiment
├── plots/ # All generated plots
│ ├── loss_curves.png
│ ├── accuracy_curves.png
│ ├── lambda_max.png
│ ├── sharpness.png
│ └── gen_gap_vs_lambda.png
└── latex/ # LaTeX report


---

## 📦 Installation

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
torch
torchvision
numpy
matplotlib
tqdm


---

## 🧠 Conceptual Overview

This project connects three major aspects of deep learning:

### 1. Loss Landscape Geometry
- Hessian spectrum  
- Largest eigenvalue λ_max  
- Sharpness under random perturbations  
- 1D/2D loss slices  
- Effective dimensionality of curvature  

### 2. Stochastic Optimization Dynamics
- SGD as stochastic differential equation  
- Gradient noise acts as “temperature”  
- High-noise → flatter minima  
- Low-noise → sharp minima  

### 3. Generalization Behavior
- Generalization gap (train vs test)  
- Strong correlation between λ_max and generalization  
- Flat minima consistently generalize better  

---

## 🧪 Running Experiments

### Train all four experiments (MLP/CNN × high/low noise)


Logs saved under:


Logged metrics:

- epoch  
- train/val/test loss  
- train/val/test accuracy  
- λ_max (largest Hessian eigenvalue)  
- sharp_mean / sharp_max  

Also stores:

loss_slice_2d.pt


Generated output:

- loss_curves.png  
- accuracy_curves.png  
- lambda_max.png  
- sharpness.png  
- gen_gap_vs_lambda.png  

---

## 📉 Summary of Experimental Findings

### ✔ Loss Curves
CNNs converge faster and achieve lower loss.  
High-noise SGD smooths early optimization.

### ✔ Accuracy Curves
CNN reaches >99% rapidly.  
High-noise improves early generalization.

### ✔ Curvature (λ_max)
- CNN low-noise → λ_max > 100 (very sharp)  
- CNN high-noise → λ_max ≈ 5–15 (flat)  
- MLPs follow same trend at smaller scale  

### ✔ Sharpness
Matches λ_max trends across all runs.

### ✔ Generalization vs Curvature
Flatter minima → smaller generalization gap  
Sharp minima → worse generalization  
Matches results from Keskar et al. (2017).  

---

## 🔮 Extensions to LLMs & OCR Systems

### LLM Training / Fine-tuning:
- Track curvature spikes for instability detection  
- Identify sharp layers (attention blocks)  
- Guide LR warmup/decay, clipping, and LoRA tuning  
- High-noise fine-tuning improves generalization  

### OCR Models:
- CNN/Vision Transformer encoders show different curvature profiles  
- Sharp minima = overfitting to noisy scans  
- Use curvature to choose robust model/encoder  

---

## 📄Research Report

The `latex/` folder includes a full scientific paper template with:
- Theory  
- Methods  
- Results  
- Figures  
- Interpretation  
- Application to LLM/OCR  

Ready to compile in Overleaf.

---

## 🤝 Contributing

Possible extensions:

- Add ResNet/Vision Transformer experiments  
- Layer-wise curvature tracking  
- Fisher information-based curvature  
- Optimization trajectory visualization  

