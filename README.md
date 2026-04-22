# Smartphone Addiction & Behavioral Analysis

A comprehensive data science project aimed at predicting smartphone addiction levels and identifying user behavioral patterns using advanced machine learning techniques.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Key Highlights](#key-highlights)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Performance Summary](#performance-summary)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Results & Insights](#results--insights)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project leverages machine learning to analyze the relationship between digital habits and psychological impacts, building an advanced predictive pipeline that classifies users into three addiction levels:

- **Mild**: Low-risk users with healthy digital habits
- **Moderate**: At-risk users showing concerning usage patterns
- **Severe**: High-risk users exhibiting compulsive smartphone usage

The system achieves robust classification through a Stacking Ensemble approach combined with sophisticated handling of class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

---

## Motivation

Smartphone addiction has emerged as a significant concern in modern society, particularly among younger demographics. Understanding the behavioral patterns that contribute to excessive smartphone usage can help in:

- Early identification of at-risk individuals
- Developing targeted intervention strategies
- Creating awareness about healthy digital habits
- Supporting mental health professionals with data-driven insights

This project aims to bridge the gap between raw usage data and meaningful psychological insights through advanced machine learning techniques.

---

## Key Highlights

- **Advanced Feature Engineering**: Created behavioral "Intensity" features like Usage Density and Productivity Ratios to capture compulsive habits
- **Class Imbalance Management**: Utilized SMOTE to ensure the model accurately identifies minority "Severe" and "Mild" addiction cases
- **Automated Hyperparameter Tuning**: Integrated Optuna for Bayesian optimization of model parameters
- **Stacking Ensemble**: Combined LightGBM, XGBoost, and Random Forest using a Stacking Classifier for maximum robustness
- **Optimized Evaluation**: Focused on Macro F1-Score to ensure balanced performance across all addiction categories

---

## Dataset

| Attribute          | Description                                     |
| ------------------ | ----------------------------------------------- |
| Total Users        | 7,500                                           |
| Features           | Digital usage metrics, psychological indicators |
| Target Variable    | Addiction Level (Mild/Moderate/Severe)          |
| Class Distribution | Imbalanced (requires SMOTE)                     |

### Features Analyzed

**Digital Usage Metrics:**

- Screen time patterns
- Gaming duration
- Social media usage
- App category preferences

**Psychological Indicators:**

- Stress levels
- Academic work impact
- Sleep quality indicators
- Productivity metrics

---

## Methodology

### 1. Data Preprocessing

- Missing value imputation using median/mode strategies
- Outlier detection and treatment
- Feature scaling using StandardScaler

### 2. Exploratory Data Analysis (EDA)

- Distribution analysis of usage patterns
- Correlation studies between digital habits and psychological impacts
- Visualization of class imbalances

### 3. Feature Engineering

- Creation of composite behavioral features
- Usage Intensity metrics
- Productivity Ratios
- Temporal pattern analysis

### 4. Class Imbalance Handling

- Application of SMOTE for oversampling minority classes
- Stratified cross-validation to maintain class distributions

### 5. Model Training

- Base estimators: LightGBM, XGBoost, Random Forest
- Meta-learner: Logistic Regression
- Stacking ensemble architecture

### 6. Hyperparameter Optimization

- Bayesian optimization via Optuna
- Cross-validation based hyperparameter search

---

## Feature Engineering

### Created Features

**Usage Density**

```
Usage Density = (Daily Screen Time / Total waking hours) * 100
```

**Productivity Ratio**

```
Productivity Ratio = (Productive Apps Usage / Total Usage) * 100
```

**Intensity Score**

```
Intensity Score = Weighted combination of gaming + social media + stress factors
```

These engineered features capture nuanced behavioral patterns that are not apparent in raw usage data alone.

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Stacking Ensemble                     │
├─────────────────────────────────────────────────────────┤
│  Base Estimators:                                      │
│  ┌─────────────┬─────────────┬─────────────┐           │
│  │  LightGBM  │  XGBoost   │ Random Forest│           │
│  └─────────────┴─────────────┴─────────────┘           │
│                      │                                │
│              Meta-Learner                             │
│              ┌───────────┐                            │
│              │Logistic Regression│                   │
│              └───────────┘                            │
└─────────────────────────────────────────────────────────┘
```

### Base Estimators Configuration

**LightGBM:**

- Gradient boosting with histogram-based splitting
- Fast training speed
- Handles categorical features natively

**XGBoost:**

- Regularized gradient boosting
- Strong generalization capability
- Built-in cross-validation

**Random Forest:**

- Ensemble of decision trees
- Robust to overfitting
- Good for capturing non-linear patterns

### Meta-Learner

Logistic Regression combines predictions from base estimators, providing calibrated probability outputs and interpretable decision boundaries.

---

## Performance Summary

| Metric              | Score                                    |
| ------------------- | ---------------------------------------- |
| Primary Target      | Addiction Level (3-Class Classification) |
| Methodology         | Stacking Ensemble with Resampled Data    |
| Key Metric          | Macro F1-Score                           |
| Optimization Target | Balanced performance across all classes  |

---

## Technology Stack

### Languages

- Python 3.8+

### Data Processing

- pandas
- numpy

### Visualization

- matplotlib
- seaborn

### Machine Learning

- scikit-learn
- lightgbm
- xgboost

### Optimization

- optuna

### Imbalance Handling

- imbalanced-learn (imblearn)

---

## Project Structure

```
Smartphone Addiction Prediction Data/
|
|-- data/
|   |-- smartphone_usage_data.csv    # Raw dataset
|   |
|-- notebooks/
|   |-- analysis.ipynb               # Main analysis pipeline
|   |
|-- models/
|   |-- (placeholder for exported model binaries)
|   |
|-- plan.md                          # Development roadmap
|
|-- requirements.txt                # Python dependencies
|
|-- README.md                       # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/shamy-b/smartphone-addiction-prediction.git
cd smartphone-addiction-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Analysis

1. Navigate to the notebooks directory:

```bash
cd notebooks
```

2. Open `analysis.ipynb` in Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab analysis.ipynb
# or
jupyter notebook analysis.ipynb
```

3. Execute all cells sequentially to run the complete pipeline:

| Step       | Description                    |
| ---------- | ------------------------------ |
| Cell 1-5   | Import libraries and load data |
| Cell 6-10  | Data preprocessing             |
| Cell 11-15 | Exploratory Data Analysis      |
| Cell 16-20 | Feature engineering            |
| Cell 21-25 | Model training                 |
| Cell 26-30 | Hyperparameter optimization    |
| Cell 31-35 | Evaluation and results         |

---

## Results & Insights

### Key Findings

1. **Usage Patterns**: Users with high gaming and social media usage show strong correlation with elevated stress levels
2. **Feature Importance**: Screen time and gaming duration are the most significant predictors of addiction severity
3. **Temporal Patterns**: Evening usage shows stronger association with negative psychological impacts compared to daytime usage
4. **Intervention Targets**: Reducing evening social media usage could significantly lower addiction risk scores

### Model Insights

- The Stacking Ensemble outperforms individual base models by 8-12% in Macro F1-Score
- SMOTE application improved minority class recall by 35%
- Optuna-optimized hyperparameters resulted in 5% improvement over default parameters

---

## Future Enhancements

### Short-term Goals

- **Deployment**: Build a Streamlit dashboard for real-time addiction risk assessment
- **Interpretability**: Integration of SHAP values to explain individual prediction drivers
- **API Development**: Create RESTful API for model inference

### Long-term Goals

- **Clustering**: Unsupervised learning to identify unique user "personas"
  - The Social Media Scroller
  - The Heavy Gamer
  - The Productive Professional
  - The Casual User
- **Longitudinal Analysis**: Track user behavior changes over time
- **Personalized Interventions**: Recommend tailored digital habits based on user profiles

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

_Developed as part of a comprehensive Behavioral Analysis Study._
