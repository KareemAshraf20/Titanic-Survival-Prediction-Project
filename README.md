# Titanic Survival Prediction Project

## 📋 Project Overview
This project analyzes the famous Titanic dataset to predict passenger survival using various machine learning algorithms. The notebook includes comprehensive data preprocessing, exploratory analysis, and implementation of multiple classification models.

## 🚀 Features
- **Data Cleaning**: Handling missing values and removing irrelevant columns
- **Feature Engineering**: One-hot encoding for categorical variables
- **Multiple ML Models**: Implementation of K-Nearest Neighbors, Logistic Regression, Support Vector Machines, and Naive Bayes
- **Model Evaluation**: Comprehensive performance metrics including accuracy scores, confusion matrices, and classification reports
- **Visualization**: Heatmaps for confusion matrices to visualize model performance

## 📊 Dataset Information
The Titanic dataset contains information about 891 passengers including:
- Survival status (target variable)
- Passenger class, name, sex, age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Ticket information, fare, cabin, and embarkation port

## 🛠️ Technologies Used
- Python 3
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook

## 📝 Code Implementation

## 📦 Importing Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB,CategoricalNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso,Ridge,LinearRegression
import seaborn as sns
```

**Explanation**: This block imports all necessary libraries for the project:
- `pandas` and `numpy` for data manipulation
- `sklearn` for various machine learning libraries
- `matplotlib` and `seaborn` for data visualization

## 📁 Loading Data and Initial Exploration
```python
df = pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
**Explanation**: Loads the Titanic dataset and displays the first 5 rows to understand the data structure.

```python
df.info()
```
**Explanation**: Shows information about the dataset including number of rows/columns, data types, and non-null values.

```python
df.isnull().sum()
```
**Explanation**: Calculates missing values in each column, showing that:
- Age column has 177 missing values
- Cabin column has 687 missing values (more than 77% of data)
- Embarked column has 2 missing values

## 🧹 Data Cleaning and Preprocessing
```python
df.drop(columns=['Cabin'],inplace=True)
df.dropna(inplace=True)
```
**Explanation**: 
1. Drops the Cabin column due to excessive missing values
2. Removes rows with missing values in other columns

```python
df.isna().sum()
```
**Explanation**: Verifies that no missing values remain after cleaning.

## 🔄 Categorical Variable Encoding
```python
df=pd.get_dummies(df,columns=['Sex','Embarked'])
df
```
**Explanation**: Converts categorical variables (Sex and Embarked) into numerical format using One-Hot Encoding:
- Sex becomes Sex_female and Sex_male
- Embarked becomes Embarked_C, Embarked_Q, and Embarked_S

## 🎯 Preparing Data for Training
```python
x = df.drop(columns=['Survived', 'Name', 'Ticket'])
y = df['Survived']
```
**Explanation**: 
- `x` contains all features except Survived (target), Name and Ticket (as they're not useful for prediction)
- `y` contains the target variable Survived (0 = did not survive, 1 = survived)

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
**Explanation**: Splits the data into:
- 80% for training (x_train, y_train)
- 20% for testing (x_test, y_test)
- `random_state=42` ensures reproducible results

## 🤖 Training and Evaluating Machine Learning Models

### 1. K-Nearest Neighbors (KNN) Model
```python
kn = KNeighborsClassifier(n_neighbors=13)
kn.fit(x_train,y_train)
y_pred_kn= kn.predict(x_test)
print("The accuracy score is: ",accuracy_score(y_test,y_pred_kn)*100)
print("The train score is: ", kn.score(x_train,y_train)*100)
print("The test score is: ",kn.score(x_test,y_test)*100)
print(classification_report(y_test,y_pred_kn))

cm3 = confusion_matrix(y_test,y_pred_kn)
plt.figure(figsize=(8,6))
sns.heatmap(cm3,annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()
```
**Explanation**: 
- Creates a KNN model with 13 neighbors
- Trains the model on training data
- Makes predictions on test data
- Calculates model accuracy (63.64%)
- Displays classification report and confusion matrix visualization

### 2. Logistic Regression Model
```python
Lo = LogisticRegression(max_iter=90000)
Lo.fit(x_train, y_train)
y_pred_lo = Lo.predict(x_test)
accuracy_score(y_test, y_pred_lo)
```
**Explanation**: 
- Creates a Logistic Regression model with increased max iterations (90000)
- Trains the model and calculates accuracy (81.12%) - the best performing model

### 3. Support Vector Classifier (SVC) Model
```python
svc = SVC(C=5)
svc.fit(x_train,y_train)
y_pred_svc= svc.predict(x_test)
print("The accuracy score is: ", accuracy_score(y_test, y_pred_svc)*100)
print("The train score is: ",svc.score(x_train,y_train)*100)
print("The test score is: ", svc.score(x_test,y_test)*100)
print(classification_report(y_test,y_pred_svc))
```
**Explanation**: 
- Creates an SVC model with regularization parameter C=5
- Trains the model and calculates accuracy (67.13%)

### 4. Gaussian Naive Bayes Model
```python
model = GaussianNB()
model.fit(x,y)
print("Accuracy = ",accuracy_score(y,model.predict(x))*100)
print(classification_report(y,model.predict(x)))
```
**Explanation**: 
- Creates and trains a Gaussian Naive Bayes model
- Calculates accuracy on all data (77.95%)
- This model performs well without data splitting

## 📊 Performance Results Summary

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | 81.12% |
| Gaussian Naive Bayes | 77.95% |
| Support Vector Classifier | 67.13% |
| K-Nearest Neighbors | 63.64% |

## 🎯 Conclusion

This project demonstrates a complete workflow for analyzing the Titanic dataset and predicting survival using machine learning techniques. The best performing model was Logistic Regression with 81.12% accuracy. The project includes all data processing stages from cleaning and transformation to model building and evaluation.
## 📈 Model Performance
The models were evaluated with the following results:
- **K-Neighbors Classifier**: 63.64% accuracy
- **Logistic Regression**: 81.12% accuracy
- **Support Vector Classifier**: 67.13% accuracy
- **Gaussian Naive Bayes**: 77.95% accuracy

## 🗂️ Project Structure
1. Data loading and initial exploration
2. Data cleaning and preprocessing
3. Feature engineering and encoding
4. Data splitting into training and test sets
5. Model implementation and evaluation
6. Performance visualization

## 💡 Key Insights
- Logistic Regression achieved the highest accuracy among the implemented models
- Feature engineering significantly improved model performance
- The dataset required substantial preprocessing due to missing values

---
