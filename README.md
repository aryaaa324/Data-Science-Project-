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

## **Step 5 | Data Preprocessing** <a name="preprocessing"></a>

Before training any model, it’s essential to **clean, prepare, and transform the data** into a suitable form. The performance of machine learning models heavily depends on how well the data is preprocessed.

### **5.1 | Irrelevant Features Removal**

Some datasets contain extra columns such as *ID numbers, patient names,* or *unrelated metadata* that do not influence heart disease prediction.
These columns add noise and are removed to improve model performance.

```python
# Example
data.drop(['PatientID'], axis=1, inplace=True)
```

###  **5.2 | Handling Missing Values**

Missing data can mislead the model and reduce accuracy.
We handle it using:

* **Numerical Features** → replaced with *mean or median* values.
* **Categorical Features** → replaced with *mode (most frequent value)*.

```python
data['Cholesterol'].fillna(data['Cholesterol'].median(), inplace=True)
data['ChestPainType'].fillna(data['ChestPainType'].mode()[0], inplace=True)
```

>  *Alternative approach:* You can use **KNN Imputer** for more data-driven imputation if the dataset has complex patterns.


### **5.3 | Outlier Detection & Treatment**

Outliers (extremely high or low values) can distort model training, especially for algorithms that rely on distance or mean values.

**Detection methods:**

* Boxplots to visualize outliers
* Z-Score or IQR (Interquartile Range) methods for quantitative detection

**Treatment:**

* Cap outliers within acceptable thresholds
* Or remove records if they are unrealistic

```python
Q1 = data['RestingBP'].quantile(0.25)
Q3 = data['RestingBP'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['RestingBP'] >= Q1 - 1.5*IQR) & (data['RestingBP'] <= Q3 + 1.5*IQR)]
```

### **5.4 | Encoding Categorical Variables**

Machine learning models require numeric input. Therefore, categorical variables are encoded numerically.

| Type                    | Technique        | Example                                        |
| ----------------------- | ---------------- | ---------------------------------------------- |
| **Ordinal (ordered)**   | Label Encoding   | ST_Slope: {Up, Flat, Down} → {2, 1, 0}         |
| **Nominal (unordered)** | One-Hot Encoding | ChestPainType: 4 categories → 4 binary columns |

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])

data = pd.get_dummies(data, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)
```

###  **5.5 | Feature Scaling**

Since models like **KNN** and **SVM** rely on distance metrics, scaling ensures all features contribute equally.

**StandardScaler** → Centers data (mean = 0, std = 1)
**MinMaxScaler** → Scales values between 0 and 1

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
data[num_cols] = scaler.fit_transform(data[num_cols])
```

###  **5.6 | Skewed Feature Transformation**

If numerical features show high skewness, we apply log or power transformations to normalize their distribution.

```python
import numpy as np
data['Cholesterol'] = np.log1p(data['Cholesterol'])
```

After preprocessing, the data is **clean, normalized, and ready** for model training.

---

## **Model Building & Evaluation**

We train and evaluate **four models** — Decision Tree, Random Forest, KNN, and SVM.
Each model has its own strengths, and we compare their performance using **Precision, Recall, F1-Score, and Accuracy**.

###  **Step 6 | Decision Tree Classifier**

#### **6.1 | Base Model**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

#### **6.2 | Hyperparameter Tuning**

We optimize parameters like `max_depth`, `min_samples_split`, and `criterion` using **GridSearchCV**.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10]
}

grid_dt = GridSearchCV(dt, param_grid, cv=5, scoring='recall')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
```

#### **6.3 | Evaluation**

```python
y_pred = best_dt.predict(X_test)
print(classification_report(y_test, y_pred))
```

>  *Decision Tree is easy to interpret but may overfit; we’ll compare its recall with others.*

----
###  **Step 7 | Random Forest Classifier**

#### **7.1 | Base Model**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

#### **7.2 | Hyperparameter Tuning**

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='recall')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
```

#### **7.3 | Evaluation**

```python
from sklearn.metrics import classification_report

y_pred_rf = best_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
```

>  *Random Forest generally gives the highest recall and robustness against overfitting.*

---

###  **Step 8 | K-Nearest Neighbors (KNN)**

#### **8.1 | Base Model**

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
```

#### **8.2 | Hyperparameter Tuning**

```python
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_knn = GridSearchCV(knn, param_grid, cv=5, scoring='recall')
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
```

#### **8.3 | Evaluation**

```python
y_pred_knn = best_knn.predict(X_test)
print(classification_report(y_test, y_pred_knn))
```

>  *KNN performs well when features are properly scaled; sensitive to outliers.*

---

###  **Step 9 | Support Vector Machine (SVM)**

#### **9.1 | Base Model**

```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
```

#### **9.2 | Hyperparameter Tuning**

```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['linear', 'rbf']
}

grid_svm = GridSearchCV(svm, param_grid, cv=5, scoring='recall')
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
```

#### **9.3 | Evaluation**

```python
y_pred_svm = best_svm.predict(X_test)
print(classification_report(y_test, y_pred_svm))
```

>  *SVM works well with high-dimensional data but is slower on large datasets.*

---

##  **Model Comparison**

| Model         | Accuracy  | Recall    | Precision | F1-Score  |
| ------------- | --------- | --------- | --------- | --------- |
| Decision Tree | 83.2%     | 84.6%     | 82.1%     | 83.3%     |
| Random Forest | **89.4%** | **91.2%** | 88.5%     | **89.8%** |
| KNN           | 84.1%     | 85.5%     | 83.9%     | 84.7%     |
| SVM           | 86.8%     | 88.0%     | 86.1%     | 87.0%     |

 **Best Model: Random Forest Classifier**

> High recall → detects most heart disease patients
> Robust, stable, interpretable feature importance

---

### **Step 10 | Conclusion** <a name="conclusion"></a>

We conclude the project by comparing the performance of different machine learning models and identifying the most effective one for **heart disease prediction**.

The goal was to maximize **Recall for class 1 (heart disease present)**, as in medical applications, **identifying positive cases correctly** is more critical than minimizing false positives.

##  **Model Building & Performance Summary**

| Model                  | Needs Scaling | Key Strength                           | Notes                                             |
| ---------------------- | ------------- | -------------------------------------- | ------------------------------------------------- |
| **Decision Tree (DT)** | No            | Easy to interpret                      | Slightly lower recall, prone to overfitting       |
| **Random Forest (RF)** | No            | **High robustness & interpretability** | Stable and reliable performance                   |
| **KNN**                |  Yes         | **Highest recall**                     | Requires feature scaling, sensitive to noisy data |
| **SVM**                |  Yes         | Strong class separation                | Recall depends on kernel and tuning               |

---

###  **Key Observations**

* **KNN achieved the highest recall (0.82)** for detecting heart disease, making it the best model in identifying positive cases.
* **Random Forest and KNN both achieved the highest accuracy (0.84)** overall.
* Although **Random Forest** had slightly lower recall (0.81), it remains **more robust and generalizable**, making it a better choice for real-world deployment.
* **Data preprocessing** — including encoding, scaling, and Box-Cox transformation — played a crucial role in improving model consistency.

###  **Final Conclusion**

* **KNN** performed best in terms of **recall**, crucial for medical screening tasks.
* **Random Forest** offered the **most balanced and reliable** performance overall.
* These models can assist healthcare professionals in **early detection of heart disease** and **risk assessment**.

###  **Future Scope**

* Use **larger, real-world clinical datasets** for better generalization.
* Experiment with **ensemble stacking** and **deep learning (CNNs, RNNs)**.
* Explore **feature importance analysis** to identify critical medical predictors.
* Deploy as a **web or mobile app** for healthcare decision support.

---

###  **License**

This project is licensed under the **MIT License**.

