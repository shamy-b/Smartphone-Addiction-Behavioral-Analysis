# 📱 Smartphone Addiction & Behavioral Analysis

A comprehensive data science project aimed at predicting smartphone addiction levels and identifying user behavioral patterns using advanced machine learning techniques.

## 🚀 Project Overview
This project leverages a dataset of 7,500 users to analyze the relationship between digital habits (screen time, gaming, social media) and psychological impacts (stress, academic work impact). We built an advanced predictive pipeline that classifies users into **Mild**, **Moderate**, or **Severe** addiction levels.

### Key Highlights:
- **Advanced Feature Engineering**: Created behavioral "Intensity" features like Usage Density and Productivity Ratios to capture compulsive habits.
- **Class Imbalance Management**: Utilized SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model accurately identifies minority "Severe" and "Mild" addiction cases.
- **Automated Hyperparameter Tuning**: Integrated Optuna for Bayesian optimization of model parameters.
- **Stacking Ensemble**: Combined the predictive power of LightGBM, XGBoost, and Random Forest using a Stacking Classifier for maximum robustness.

## 📊 Performance Summary
- **Primary Target**: Addiction Level (3-Class Classification)
- **Methodology**: Stacking Ensemble with Resampled Data
- **Key Metric**: Optimized for Macro F1-Score to ensure balanced performance across all addiction categories.

## 🛠️ Technology Stack
- **Languages**: Python
- **Libraries**: 
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `lightgbm`, `xgboost`
  - Optimization: `optuna`
  - Imbalance Handling: `imbalanced-learn`

## 📁 Project Structure
- `data/`: Contains the raw usage dataset.
- `notebooks/`: 
  - `analysis.ipynb`: The main end-to-end pipeline (Preprocessing -> EDA -> Modeling -> Optimization).
- `models/`: Placeholder for exported model binaries.
- `plan.md`: The original development roadmap.

## 🏁 Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Analysis Notebook:
   Navigate to `notebooks/analysis.ipynb` and execute all cells to reproduce the modeling and evaluation.

## 📈 Future Enhancements
- **Deployment**: Plans to build a Streamlit dashboard for real-time addiction risk assessment.
- **Clustering**: Unsupervised learning to identify unique user "personas" (e.g., The Social Media Scroller vs. The Heavy Gamer).
- **Interpretability**: Integration of SHAP values to explain individual prediction drivers.

---
*Developed as part of a comprehensive Behavioral Analysis Study.*
