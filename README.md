# Pneumonia Detection: Traditional ML vs Deep Learning

## GCGO Healthcare - Optimizing and Reducing Errors in Critical Diagnosis

A comprehensive comparative study evaluating traditional machine learning and deep learning approaches for automated pneumonia detection from chest X-ray images.

---

## Project Overview

**Research Objective:** Compare performance of traditional ML (handcrafted features) vs deep learning (automatic feature learning) for pneumonia detection to inform clinical deployment decisions in resource-constrained healthcare settings.

**Problem Context:**
- 2.5 million annual pneumonia deaths globally (610,000 children under 5)
- Radiologist inter-observer variability: 20-56%
- Diagnostic error rates: 60-80% perceptual errors
- Clinical impact: Missed diagnoses lead to delayed treatment and preventable mortality

**Mission:** Reduce diagnostic errors through automated, accurate, and accessible diagnostic support systems.

---

##  Key Findings

| Approach | Best Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|----------|-----------|----------|-----------|--------|----------|---------------|
| **Traditional ML** | SVM (RBF) | **98.34%** | 99.48% | 98.28% | 0.9888 | 31.6s |
| **Deep Learning** | CNN + Dropout | 97.96% | 98.63% | 98.63% | 0.9863 | ~45 min |

**Significance:** Both approaches substantially exceed radiologist performance (70-85% agreement), with traditional ML offering superior accuracy and practical deployment advantages.

---

##  Methodology

### Traditional Machine Learning (7 Models)
- **Feature Extraction:** HOG, LBP, GLCM, Statistical Features (373 dimensions)
- **Preprocessing:** StandardScaler + PCA (95% variance)
- **Models:** Logistic Regression, SVM (Linear/RBF), Random Forest, KNN, Gradient Boosting, Naive Bayes

### Deep Learning (5 Models)
- **Architectures:** Custom CNNs, Transfer Learning (MobileNetV2, ResNet50)
- **Configuration:** 128×128 input, Adam optimizer, binary cross-entropy loss
- **Training:** 10-15 epochs with early stopping and learning rate scheduling

### Dataset
- **Source:** Kaggle Chest X-ray Pneumonia Dataset
- **Total Images:** 5,863 (1,583 Normal, 4,273 Pneumonia)
- **Patient Age:** Pediatric (1-5 years)
- **Split:** 85% train, 15% validation, separate test set

---

## Quick Start

### Prerequisites
```bash
pip install tensorflow scikit-learn scikit-image opencv-python pandas numpy matplotlib seaborn kagglehub tqdm
```

### Usage
1. Open `Pneumonia_ML_vs_DL_Complete_Analysis.ipynb` in Google Colab or Jupyter
2. Enable GPU (optional, recommended for deep learning): Runtime → Change runtime type → GPU
3. Run all cells sequentially
4. Dataset downloads automatically via kagglehub (~1.2 GB)
5. Total execution time: ~50 minutes (with GPU) or ~2 hours (CPU only)


##  Results Summary

### Traditional ML Performance
- **Best:** SVM-RBF (98.34% accuracy, 31.6s training)
- **Fastest:** Logistic Regression (97.32% accuracy, 1.7s training)
- **Most Efficient:** KNN (95.53% accuracy, 0.03s training)

### Deep Learning Performance
- **Best:** CNN + Dropout (97.96% accuracy)
- **Consistent:** All models 96.8-98.0% accuracy
- **Advantage:** No manual feature engineering required

### Clinical Significance
- **Recall Rates:** 98.28% (SVM-RBF), 98.63% (CNN+Dropout)
- **Interpretation:** <2% missed pneumonia cases vs 15-30% in current practice
- **Impact:** Could prevent thousands of deaths if deployed in high-burden regions

---

## Deployment Recommendations

### For Resource-Constrained Settings
 **Deploy Traditional ML (SVM-RBF)**
- No GPU required
- Fast training (<1 min) and inference
- Small model size (offline deployment)
- Interpretable decisions
- State-of-the-art accuracy (98.34%)

### For Research Hospitals
 **Consider Deep Learning (CNN + Dropout)**
- Requires GPU infrastructure
- End-to-end learning (no feature engineering)
- Extensible to multi-task platforms
- Robust performance (97.96%)

---
---

**GCGO:** GCGO Healthcare  
**Mission:** Optimizing and Reducing Errors Associated with Critical Diagnosis For Better and Efficient Patient Treatment

---

##  Acknowledgments

- Dataset: Kermany et al., Kaggle Chest X-ray Pneumonia Dataset
- Computational resources: Google Colab
- Framework: TensorFlow, scikit-learn, scikit-image

---

