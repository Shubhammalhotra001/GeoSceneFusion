## GeoSceneFusion

This project implements a hybrid deep learning pipeline to **classify satellite images** into **Urban** and **Rural** categories. It combines transfer learning on **EfficientNet‑B0** with a custom CNN head to achieve strong performance on the **NWPU‑RESISC45** benchmark.

## 📌 Project Goal

Automatically distinguish urban from rural land covers in satellite imagery to support **urban planning**, **environmental monitoring**, and **smart agriculture**.

## ⚙️ Tech Stack

- **Python**  
- **PyTorch**  
- **Torchvision**  
- **NumPy**  
- **Matplotlib**  
- **scikit-learn**  
- **Google Colab** (for training)

## 🧠 Model Architecture

1. **EfficientNet‑B0** (ImageNet‑pretrained)  
2. **Custom CNN** head (dropout‑regularized global pooling + dense layers)  
3. **Concatenation** of features + final softmax classifier

## 📁 Dataset

- **NWPU‑RESISC45** (45 scene classes, 31,500 RGB images)  
  - Official homepage & manual download:  
    http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html :contentReference[oaicite:0]{index=0}  
  - Direct OneDrive download (RAR):  
    https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs :contentReference[oaicite:1]{index=1}  
  - Kaggle mirror:  
    https://www.kaggle.com/datasets/aqibrehmanpirzada/nwpuresisc45 :contentReference[oaicite:2]{index=2}  

- **Classes merged**:  
  - *Urban*: Airport, Industrial, Highway, …  
  - *Rural*: Pasture, Forest, Annual Crop, …

## 🚀 How to Run

1. **Clone**  
   ```bash
   git clone https://github.com/Shubhammalhotra001/GeoSceneFusion.git
   train_resisc45.py
## Install dependencies
pip install -r requirements.txt

📊 Results
Peak Validation Accuracy: 84.84 %

F1‑Score ≥ 0.80 for 38 out of 45 classes

Inference Speed: > 50 fps on NVIDIA GTX 1650
