# Heart Disease Prediction Using Machine Learning

## Introduction
Heart disease is one of the leading causes of mortality globally. Early detection is crucial to preventing life-threatening cardiac events. This project analyzes patient medical attributes such as **age, blood pressure, cholesterol, heart rate, chest pain type**, etc., to predict the likelihood of heart disease.

The primary goal is to **build a predictive machine learning model** with **high recall for class 1 (patients with heart disease)** to ensure we **do not miss diagnosing a patient at risk**.

---
## Motivation
Traditional diagnosis methods often rely on a combination of clinical judgment, ECG patterns, and lab results. However, these can be subjective or require expensive medical tests.
With the growth of machine learning (ML) in healthcare, it is now possible to build automated systems that:
- Detect potential heart disease at an early stage
- Provide risk scores for medical professionals
- Reduce diagnostic costs
- Enhance preventive care
> By leveraging predictive analytics, this system assists in early intervention and decision-making.

##  Problem Statement
Develop a machine learning model that predicts whether a patient is at risk of heart disease based on medical measurements, focusing on **maximizing recall** to identify as many heart-disease patients as possible.

---

##  Objectives
* **Explore the Dataset** to understand feature distributions and behavior
* Perform **Extensive EDA** — univariate and bivariate analysis with respect to the target
* Apply **Data Preprocessing**, including:
  - Removal of irrelevant attributes
  - Handling missing values and outliers
  - Encoding categorical features
  - Scaling & normalizing skewed attributes
* **Model Training** using:
  - Decision Tree
  - Random Forest
  - KNN
  - SVM
* **Optimize hyperparameters** for improved performance
* **Evaluate models** with Precision, Recall, and F1-Score (Recall prioritized)

---

##  Table of Contents  

* [Step 1 | Import Libraries](#import)
* [Step 2 | Read Dataset](#read)
* [Step 3 | Dataset Overview](#overview)
    - [Step 3.1 | Dataset Basic Information](#basic)
    - [Step 3.2 | Summary Statistics for Numerical Variables](#num_statistics)
    - [Step 3.3 | Summary Statistics for Categorical Variables](#cat_statistics)
* [Step 4 | EDA](#eda)
    - [Step 4.1 | Univariate Analysis](#univariate)
        - [Step 4.1.1 | Numerical Variables Univariate Analysis](#num_uni)
        - [Step 4.1.2 | Categorical Variables Univariate Analysis](#cat_uni)
    - [Step 4.2 | Bivariate Analysis](#bivariate)
        - [Step 4.2.1 | Numerical Features vs Target](#num_target)
        - [Step 4.2.2 | Categorical Features vs Target](#cat_target)
* [Step 5 | Data Preprocessing](#preprocessing)
    - [Step 5.1 | Irrelevant Features Removal](#feature_removal)
    - [Step 5.2 | Missing Value Treatment](#missing)
    - [Step 5.3 | Outlier Treatment](#outlier)
    - [Step 5.4 | Categorical Features Encoding](#encoding)
    - [Step 5.5 | Feature Scaling](#scaling)
    - [Step 5.6 | Transforming Skewed Features](#transform)
* [Step 6 | Decision Tree Model Building](#dt)
    - [Step 6.1 | DT Base Model Definition](#dt_base)
    - [Step 6.2 | DT Hyperparameter Tuning](#dt_hp)
    - [Step 6.3 | DT Model Evaluation](#dt_eval)
* [Step 7 | Random Forest Model Building](#rf)
    - [Step 7.1 | RF Base Model Definition](#rf_base)
    - [Step 7.2 | RF Hyperparameter Tuning](#rf_hp)
    - [Step 7.3 | RF Model Evaluation](#rf_eval)
* [Step 8 | KNN Model Building](#knn)
    - [Step 8.1 | KNN Base Model Definition](#knn_base)
    - [Step 8.2 | KNN Hyperparameter Tuning](#knn_hp)
    - [Step 8.3 | KNN Model Evaluation](#knn_eval)
* [Step 9 | SVM Model Building](#svm)
    - [Step 9.1 | SVM Base Model Definition](#svm_base)
    - [Step 9.2 | SVM Hyperparameter Tuning](#svm_hp)
    - [Step 9.3 | SVM Model Evaluation](#svm_eval)
* [Step 10 | Conclusion](#conclusion)

---

##  Dataset Overview
The dataset consists of medical records used to determine heart disease likelihood.

| Feature          | Description                                             | Type        |
| ---------------- | ------------------------------------------------------- | ----------- |
| `Age`            | Age of the patient                                      | Numeric     |
| `Sex`            | 1 = Male, 0 = Female                                    | Categorical |
| `ChestPainType`  | Type of chest pain (e.g., typical angina, asymptomatic) | Categorical |
| `RestingBP`      | Resting blood pressure (mm Hg)                          | Numeric     |
| `Cholesterol`    | Serum cholesterol (mg/dl)                               | Numeric     |
| `FastingBS`      | Fasting blood sugar >120 mg/dl                          | Binary      |
| `RestingECG`     | ECG results (normal, ST-T wave abnormality, etc.)       | Categorical |
| `MaxHR`          | Maximum heart rate achieved                             | Numeric     |
| `ExerciseAngina` | Exercise-induced angina (Yes/No)                        | Categorical |
| `Oldpeak`        | ST depression induced by exercise                       | Numeric     |
| `ST_Slope`       | Slope of peak exercise ST segment                       | Categorical |
| **`Target`**     | 1 = Heart Disease, 0 = No Heart Disease                 | Binary      |

---
##  Step-by-Step Explanation (Simple & Clear)

### **Step 1 | Import Libraries** <a name="import"></a>
We begin by importing all the required Python libraries that will help in:
- Data loading and manipulation (`pandas`, `numpy`)
- Data visualization (`matplotlib`, `seaborn`)
- Machine learning model building (`sklearn`)

These libraries provide efficient tools for analysis and model development.

---

### **Step 2 | Read Dataset** <a name="read"></a>
The dataset (`heart.csv`) is loaded into a pandas DataFrame.  
This allows us to work with the data in a structured table format and perform further analysis.

---

### **Step 3 | Dataset Overview** <a name="overview"></a>
In this step, we get an initial understanding of the dataset.

#### **3.1 Dataset Basic Information** <a name="basic"></a>
We examine:
- Total number of rows and columns
- Data types of each feature (numerical / categorical)
- Whether missing values are present

This helps us understand what type of preprocessing is needed.

#### **3.2 Summary Statistics for Numerical Variables** <a name="num_statistics"></a>
Using `.describe()`, we get:
- Minimum and maximum values
- Mean and median values
- Standard deviation (spread of data)

This helps identify irregularities, skewness, and potential outliers.

#### **3.3 Summary Statistics for Categorical Variables** <a name="cat_statistics"></a>
We analyze the frequency of unique categories in features such as:
- Chest pain type
- Exercise-induced angina
- Slope, etc.

This helps understand distribution patterns in categorical variables.

---

### **Step 4 | Exploratory Data Analysis (EDA)** <a name="eda"></a>

#### **4.1 Univariate Analysis** <a name="univariate"></a>
This involves studying each feature independently.

- **Numerical Features** (e.g., age, cholesterol, blood pressure)  
  → Visualized using **histograms, density plots**
  
- **Categorical Features** (e.g., gender, chest pain type)  
  → Visualized using **count plots / bar charts**

This helps to understand data distribution and detect skewness.

#### **4.2 Bivariate Analysis** <a name="bivariate"></a>
Here we analyze how features relate to the **target variable** (Heart Disease).

- **Numerical vs Target**  
  → Visualized using **boxplots / violin plots**
  
- **Categorical vs Target**  
  → Visualized using **stacked bars / grouped count plots**

This step highlights which features are impactful predictors.

---

### **Step 5 | Data Preprocessing** <a name="preprocessing"></a>
Before training models, we clean and transform the data.

Steps include:
- **Removing irrelevant or duplicate columns**
- **Handling missing values** appropriately
- **Outlier treatment** to reduce distortion
- **Encoding categorical variables** (Label Encoding / One-Hot Encoding)
- **Scaling numerical variables** for models sensitive to value range
- **Applying transformations** if a feature is highly skewed

This ensures the dataset is clean and suitable for model training.

---

### **Step 6–9 | Model Building & Evaluation**

We develop and test **four machine learning models**:

| Model | Description |
|------|-------------|
| **Decision Tree** | Simple & interpretable model that splits data into decision rules |
| **Random Forest** | Ensemble of decision trees that improves accuracy and reduces overfitting |
| **KNN (K-Nearest Neighbors)** | Predicts class based on similarity to nearest data points |
| **SVM (Support Vector Machine)** | Finds the best boundary that separates the classes |

For each model:
1. Define model and train on training data
2. Perform hyperparameter tuning (GridSearchCV / cross-validation)
3. Evaluate using:
   - **Confusion Matrix**
   - **Precision, Recall, F1-score**
   - **Accuracy Score**

The primary focus is on **Recall** to correctly detect heart disease cases.

---

### **Step 10 | Conclusion** <a name="conclusion"></a>
We conclude by:
- Comparing model performance
- Identifying the model with the highest recall / best predictive power
- Summarizing key EDA insights and important predictive features
- Suggesting future improvements such as:
  - Collecting more patient data
  - Trying deep learning models
  - Improving medical feature engineering

---

##  Model Building & Performance Summary

| Model | Needs Scaling | Key Strength | Notes |
|------|-------------|-------------|------|
| Decision Tree | No | Easy to interpret | Prone to overfitting |
| Random Forest | No | **High recall & robustness** | Best overall model |
| KNN | Yes | Simple, distance-based | Highly scale-sensitive |
| SVM | Yes | Strong boundary classification | Recall depends on kernel selection |

**Metric Focus:**  
 Recall (Sensitivity) for class 1 (heart disease present) was prioritized.

---

##  Conclusion
- **Random Forest** performed best with **high recall**, making it suitable for medical early-warning systems.
- Proper **data preprocessing** significantly improved model reliability.
- The model can assist healthcare professionals by supporting early risk screening.

###  Future Scope:
- Larger real-world clinical datasets
- Deep learning & ensemble stacking
- deployment for clinical usage

---

##  License
This project is licensed under the **MIT License**.

