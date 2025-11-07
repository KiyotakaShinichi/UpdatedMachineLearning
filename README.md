# Diabetes Data Analysis & Machine Learning Projects

This repository contains a comprehensive set of machine learning experiments and analyses performed on a **diabetes dataset**. The goal was to explore, model, and evaluate predictive approaches to diabetes classification, along with clustering and visualization to understand underlying patterns.

---

## 🔍 Project Overview

The project includes multiple stages of data analysis and machine learning workflows:

1. **Data Preprocessing & Feature Engineering**
   - Handled categorical and numerical variables.
   - Standardization and encoding applied to prepare features for modeling.
   - Used cluster label as potential important feature.

2. **Exploratory Data Analysis (EDA) & Visualization**
   - Visualized feature distributions, correlations, and class imbalances.
   - Performed clustering analysis with **K-Means**, including **Elbow Method** and **Silhouette Score** to determine optimal clusters.
   - Generated insightful visualizations using libraries like `matplotlib` and `plotly`.

3. **Classification & Supervised Learning**
   - Applied multiple algorithms:
     - **Logistic Regression**
     - **Random Forest and other trees**
     - **XGBoost**
     - **LightGBM**
     - **CatBoost**
     - **Naive Bayes**
     - **Quadratic SVM**
     - **Neural Networks**
     - **KNN**
   - Evaluated models using metrics including:
     - Accuracy, Precision, Recall, F1-score
     - ROC-AUC
     - Cohen's Kappa, Matthews Correlation Coefficient (MCC)
     - Confusion Matrix
     - Lift charts and per-fold cross-validation accuracies
   - Hyperparameter tuning performed using:
     - **Grid Search**
     - **Optuna** optimization for automated parameter selection

4. **Ensemble Learning**
   - Implemented advanced ensembles for performance improvement:
     - **Voting Classifier**
     - **Stacking Classifier**
     - Boosting techniques (XGBoost, LightGBM, CatBoost)
   - Combined multiple models to improve predictive performance and robustness.

5. **Model Evaluation & Validation**
   - Stratified K-Fold cross-validation to assess consistency.
   - Learning curves to analyze bias-variance tradeoff.
   - Decision threshold optimization using **Youden's J statistic**.
   - Comprehensive performance reporting including per-fold results and visualizations.

---



