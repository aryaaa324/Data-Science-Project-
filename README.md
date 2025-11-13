# Data-Science-Project

***

# HMIS State-wise Health Data Science Project

## Overview

This project provides an end-to-end health data analytics workflow based on India's HMIS data for 10 major states. It demonstrates all core components of data science: data cleaning, exploratory data analysis (EDA), dimensionality reduction, clustering, classification, regression, and visualization—yielding insights for healthcare planners, researchers, and data scientists.

***

## Table of Contents

- [Project Structure](#project-structure)
- [1. Data Preprocessing](#1-data-preprocessing)
- [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [A. Univariate Analysis](#a-univariate-analysis)
  - [B. Bivariate Analysis](#b-bivariate-analysis)
  - [C. Multivariate Analysis](#c-multivariate-analysis)
- [3. Dimensionality Reduction (PCA)](#3-dimensionality-reduction-pca)
- [4. Clustering](#4-clustering)
- [5. Cluster Evaluation & Hierarchical Analysis](#5-cluster-evaluation--hierarchical-analysis)
- [6. Classification Models](#6-classification-models)
- [7. Regression Models & Comparison](#7-regression-models--comparison)
- [8. Visualizations & Outputs](#8-visualizations--outputs)
- [9. Key Insights](#9-key-insights)
- [10. How To Run](#10-how-to-run)
- [11. Credits](#11-credits)
- [12. License](#12-license)

***

## Project Structure

```
├── data/
|   └──HMIS_original_data.csv
│   └── HMIS_Cleaned.csv
├──codes/
│   └── cleaning_code.py
│   └── EDA.py
│   └── regression_classification_clustering.py
│   └── app.py
├── PPT/
├── Report/
└── README.md
```

***

## 1. Data Preprocessing

- Loaded monthly HMIS data, filtered for 10 focus states.
- Retained key features like lab tests, OPD attendance, emergencies, in-patient count, hemoglobin tests, and condom distribution.
- Cleaned column names, removed nulls, and extracted/engineered time features.
- Aggregated features to state-level monthly sums.
- Standardized all numerical columns using **StandardScaler** to enable fair comparisons and ML.

***

## 2. Exploratory Data Analysis (EDA)

### A. Univariate Analysis

- **Histograms and KDE plots** for all features using Seaborn and Matplotlib to examine distributions, skewness, and outliers.
    - Example: Distribution of lab tests, X-ray counts, OPD attendance.
- **Key Findings:**  
    - Most features are heavily right-skewed (majority low values, few large outliers).
    - KDE curves reveal distribution shapes not visible in bar-only histograms.

### B. Bivariate Analysis

- **Scatter plots** of top feature pairs to reveal potential linear or nonlinear relationships.
    - Example: Number of OPD Attendances vs. Number of In-Patient Counts.
- **Correlation Matrix and Heatmap:**
    - Computed Pearson correlation matrix using pandas and visualized via Seaborn `heatmap` for simplified feature sets.
    - Extracted and listed feature pairs with high positive correlation (r>0.60); summarized these in a correlation matrix.

### C. Multivariate Analysis

- **Pairplots (Matrix Plots):**
    - Built Seaborn pairplot matrices for top 4-5 features to view all scatterplots and featurewise distributions simultaneously.
    - This matrix helps spot collinearity, outliers, and clusters.
- **State-wise Analysis:**
    - Created state-vs-feature boxplots, barplots, and sex-ratio analysis with pie and scatter plots for demographic insight.

***

## 3. Dimensionality Reduction (PCA)

- Applied Principal Component Analysis to all scaled features.
- **Visualized** PC1 vs. PC2 with color/label by state for initial inspection of group separability.
- Used PCA-transformed features as input to clustering for improved visualization and reduced dimensionality.

***

## 4. Clustering

- **KMeans Clustering:**
    - Used Elbow Method to select optimal `k` (number of clusters).
    - Clustered scaled features; assigned each monthly state profile a cluster label.
- **Agglomerative (Hierarchical) Clustering:**
    - Applied Ward linkage; plotted dendrograms for visual hierarchy of state similarity.
- **DBSCAN:**
    - Density-based method that automatically detects outliers ("noise") and forms arbitrarily shaped clusters.

- **Cluster Visualization:**
    - Plotted PCA component space colored by cluster, with state shapes/symbols for overlayed comparison.

***

## 5. Cluster Evaluation & Hierarchical Analysis

- **Used Three Metrics:**
    - *Silhouette Score* (closer to 1 = better separation)
    - *Davies-Bouldin Index* (lower = better)
    - *Calinski-Harabasz Index* (higher = better)
- **Hierarchical Dendrogram:**
    - Visual summary of state-to-state similarities, with clusters merged by distance.

***

## 6. Classification Models

- **Target:** Cluster assignments from KMeans.
- **Models Used:** Logistic Regression, Random Forest, SVM, XGBoost.
- **Process:**
    - Stratified 70/30 train-test split, 5-fold cross-validation.
    - Evaluated accuracy, precision, recall, f1-score, confusion matrices.
- **Results:** All models performed at or near perfect accuracy, **Random Forest, SVM, and XGBoost had perfect test set scores**.

***

## 7. Regression Models & Comparison

- **Target:** In-Patient Head Count (predicted from all other features).
- **Models Used:** Linear Regression, Random Forest, XGBoost, LightGBM.
- **Metrics:** MAE, RMSE, R².
- **Visuals:**  
    - Barplots to compare model MAE, RMSE, R².
    - Feature importance barplots (where available).
    - Scatter plots of Actual vs Predicted for each regressor.
- **Findings:** Random Forest and XGBoost gave the best real-world fit. Linear Regression was perfect on this fold—possibly due to data structure. LightGBM underperformed.

***

## 8. Visualizations & Outputs

- **Univariate histograms & KDEs for all features**
- **Pairplots (scatter matrix) for all big feature sets**
- **Simplified and full correlation matrices & heatmaps**
- **State-wise and indicator-wise scatter, pie, and bar charts**
- **PCA scatterplots by state and by cluster**
- **Cluster assignment plots for KMeans, Agglomerative, DBSCAN**
- **Dendrogram of state similarity**
- **Classification confusion matrices**
- **Regression actual vs predicted scatter, error comparison bar chats, and feature importances**

***

## 9. Key Insights

- Service utilization indicators are highly correlated across states.
- State health service profiles fall into clear, distinct clusters in PCA/cluster analysis.
- Model results show cluster “types” are easily and reliably classified using ML—enabling real-time segmentation on new data.
- Random Forest and XGBoost are strong choices for volume forecasting in state health planning.
- EDA and machine learning enable robust evidence for operational decisions and policy benchmarking.

***

## 10. How To Run

1. Clone the repo and install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Place `/data/HMIS_Cleaned.csv` in the data directory.
3. Open and execute the main notebook or script.
4. All outputs (plots, models) save to `/outputs/`.

***

## 11. Credits

- Analysis by [Arya Kashikar]
- Data: [Govt. of India or official source](https://www.data.gov.in/)]
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, xgboost, lightgbm

***

## 12. License

MIT or open-source license of your choosing.

***

**For feedback, collaboration, or queries, please submit a GitHub issue or reach out to the author(s).**

***

Let me know if you’d like to expand any technical detail, add code snippets, or want `requirements.txt` and sample folder structure included!
