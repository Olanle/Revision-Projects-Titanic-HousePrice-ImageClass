# Revision-Projects-Pytorch-Sklearn-CNN-Classification-Regression
---
---
# ML Portfolio — Beginner Projects

Three end-to-end machine learning projects built from scratch as part of a structured progression toward Edge/Embedded ML Engineering. Each project was implemented independently — no tutorial code was followed. All decisions around architecture, preprocessing, and evaluation were made from first principles.

---

## Projects

| # | Project | Type | Key Result |
|---|---------|------|------------|
| 1 | CNN Image Classifier (CIFAR-10) | Computer Vision | 70.42% test accuracy |
| 2 | Titanic Survival Classifier | Binary Classification | 81% test accuracy |
| 3 | House Price Regression | Regression | Full pipeline with GridSearchCV |

---

## Project 1 — CNN Image Classifier from Scratch with PyTorch

### Overview
A Convolutional Neural Network built entirely from scratch in PyTorch to classify images from the CIFAR-10 dataset into 10 categories. No pretrained models were used. Every layer was designed, every training loop was written manually.

### Dataset
CIFAR-10 — 60,000 colour images (32x32 pixels) across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Loaded via `torchvision` with a `ToTensor()` transform.

### Architecture
```
Input (3 x 32 x 32)
→ Conv1 (32 filters, 3x3, padding=1) + ReLU + MaxPool → 32 x 16 x 16
→ Conv2 (64 filters, 3x3, padding=1) + ReLU + MaxPool → 64 x 8 x 8
→ Conv3 (128 filters, 3x3, padding=1) + ReLU + MaxPool → 128 x 4 x 4
→ Conv4 (256 filters, 3x3, padding=1) + ReLU + MaxPool → 256 x 2 x 2
→ Flatten → 1024
→ Dropout (p=0.5)
→ FC1 (1024 → 512) + ReLU
→ Dropout (p=0.5)
→ FC2 (512 → 10)
```

### Training Setup
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (lr=0.0001)
- Epochs: 20
- Batch Size: 64

### Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 70.42% |
| Final Training Loss | 0.8196 |
| Starting Training Loss | 1.9143 |

The loss curve showed smooth, consistent descent over 20 epochs — a healthy training run with no spikes or divergence.

### Key Learnings
- A 3x3 conv kernel without padding shrinks feature maps each layer — on a 32x32 image, 4 conv layers reduce dimensions to zero. `padding=1` preserves spatial size before pooling.
- The training loop core: `zero_grad → forward → loss → backward → step`
- Gradients are disabled during validation (`torch.no_grad()`) to save memory — only needed when updating weights
- 70.42% accuracy on a 10-class problem with no pretrained weights is a solid from-scratch baseline

### Stack
`PyTorch` `torchvision` `matplotlib`

---

## Project 2 — Titanic Survival Classifier

### Overview
A binary classification pipeline predicting whether a Titanic passenger survived, built independently with no tutorial guidance. Two models were trained and compared — Logistic Regression and Random Forest.

### Dataset
Titanic passenger dataset — 891 rows, 12 columns. Target variable: `Survived` (0 or 1).

### Preprocessing
- Dropped: `Cabin` (70%+ missing), `PassengerId`, `Name`, `Ticket` (no predictive signal)
- Filled `Age` missing values with column mean
- Filled `Embarked` missing values with mode
- Encoded `Sex` as binary: male=0, female=1
- One-hot encoded `Embarked` (3 categories: S, C, Q)
- 80/20 train/test split with `random_state=42`

### Results
| Model | Accuracy | F1 (Survived) | F1 (Not Survived) |
|-------|----------|---------------|-------------------|
| Logistic Regression | 81.0% | 0.76 | 0.84 |
| Random Forest | 81.0% | 0.77 | 0.84 |

Both models tied at 81% — indicating the limiting factor was data and features, not model choice.

### Confusion Matrix (Logistic Regression)
```
                Predicted No    Predicted Yes
Actual No           90               15
Actual Yes          19               55
```

### Key Learnings
- When two models tie, the bottleneck is usually the features, not the algorithm
- One-hot encoding is necessary for categorical variables with more than two values — simple integer mapping implies false ordering
- Mean imputation for `Age` is acceptable as a baseline; a smarter approach groups by `Pclass` before computing the mean
- Correlation analysis should always precede modelling to verify predictive signal exists in the data

### Stack
`pandas` `scikit-learn` `matplotlib`

---

## Project 3 — House Price Regression

### Overview
A full regression pipeline predicting house prices, covering preprocessing, feature scaling, cross-validation, and hyperparameter tuning with GridSearchCV. A critical lesson in data leakage was encountered and resolved during this project.

### Dataset
Synthetic house price dataset — 2000 rows, 10 columns. Target variable: `Price`.

### Preprocessing
Correct order of operations (data leakage lesson):
1. Drop `Id` column
2. Map `Garage` to binary (Yes=0, No=1)
3. One-hot encode `Location` and `Condition`
4. Define X and y
5. Train/test split (80/20)
6. Fit `StandardScaler` on train only, transform both train and test separately

### Data Leakage Lesson
In the first attempt, `StandardScaler` was fit on the entire dataset before splitting. This allowed test set statistics to contaminate the scaler — a form of data leakage. The fix was to fit the scaler only on training data, then use `transform()` (not `fit_transform()`) on the test set.

### Results
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | $243,242 | $279,860 | -0.007 |
| Random Forest (GridSearchCV) | — | — | -0.020 |

**Note:** Both models produced negative R² scores — not a modelling failure. Correlation analysis revealed the dataset was synthetically generated with prices assigned randomly, independent of all features. The highest feature correlation with Price was Floors at 0.056. No model can predict prices that have no relationship to the input features.

### Cross-Validation Results (Linear Regression)
```
Fold scores: [-0.014, -0.020, -0.004, -0.001, -0.008]
Mean R²: -0.009
```
Consistent negative scores across all folds confirmed the finding — not a fluke of a single split.

### GridSearchCV (Random Forest)
```
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}
Best params: {'max_depth': 5, 'n_estimators': 200}
Best score: -0.020
```

### Key Learnings
- Always run a correlation analysis before modelling — verify that signal exists before investing time in model selection
- Data leakage is a silent bug — code runs without errors but produces misleading results
- The correct preprocessing order is: clean → encode → split → scale
- `fit_transform()` on train, `transform()` only on test — never the other way around
- A more complex model does not automatically outperform a simpler one, especially on low-quality data
- Cross-validation gives a more reliable estimate than a single train/test split

### Stack
`pandas` `scikit-learn` `matplotlib`

---

## Repository Structure

```
ml-portfolio/
│
├── project1_cnn_cifar10/
│   ├── cnn_cifar10.ipynb
│   └── training_loss.png
│
├── project2_titanic/
│   └── titanic_classifier.ipynb
│
├── project3_house_price/
│   └── house_price_regression.ipynb
│
└── README.md
```

---

## Progression

These three projects are the foundation tier of a 10-project roadmap toward Edge/Embedded ML Engineering. Each project builds on the previous:

- **Project 1** established core PyTorch skills — tensors, DataLoaders, training loops, CNN architecture
- **Project 2** established tabular ML fundamentals — preprocessing, evaluation metrics, model comparison
- **Project 3** established regression pipelines — scaling, cross-validation, hyperparameter tuning, and the critical lesson of data leakage

Next: Transfer Learning for Medical Image Classification → Custom Object Detection → Model Quantization → Compression-Aware Edge Detection

---

*Built as part of a structured self-taught ML engineering curriculum. All implementation decisions made independently.*
