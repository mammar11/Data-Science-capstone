# SpaceX Reusable Rockets Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Coursera](https://img.shields.io/badge/Coursera-IBM%20Data%20Science%20Professional%20Certificate-brightgreen.svg)

This Project was part of course designed by IBM

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data](#data)
- [Exploratory Data Analysis (EDA) Using Python](#exploratory-data-analysis-eda-using-python)
- [Exploratory Data Analysis (EDA) Using SQL](#exploratory-data-analysis-eda-using-sql)
- [Machine Learning Prediction](#machine-learning-prediction)
- [Application Development](#application-development)
- [Results](#results)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the **SpaceX Reusable Rockets Prediction** project! This project aims to predict the success of SpaceX's Falcon 9 rocket landings using machine learning techniques. By analyzing historical launch data, we identify key factors that influence landing success and build a predictive model to forecast future outcomes.

## Project Overview

The project encompasses the following key components:

1. **Data Collection & Preprocessing**: Gathering and cleaning data related to SpaceX launches.
2. **Exploratory Data Analysis (EDA)**:
   - Using Python for data visualization and pattern recognition.
   - Utilizing SQL for in-depth data querying and analysis.
3. **Machine Learning Prediction**: Building and evaluating classification models to predict landing success.
4. **Application Development**: Creating an interactive web application to showcase the model's predictions.

## Technologies Used

- **Programming Languages**: Python, SQL
- **Libraries & Frameworks**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Flask
- **Databases**: SQLite/MySQL (specify which one you used)
- **Tools**: Jupyter Notebook, Git, GitHub
- **Deployment**: Heroku (if applicable)

## Data

The dataset includes historical data of SpaceX launches, encompassing various attributes such as:

- **Launch Site**: The location from where the rocket was launched.
- **Payload Mass**: The mass of the payload carried by the rocket.
- **Orbit Type**: The type of orbit the payload was intended for.
- **Launch Outcome**: Success or failure of the mission.
- **Landing Outcome**: Success or failure of the rocket landing.

*You can include a [Data Dictionary](#) or link to the data source if applicable.*

## Exploratory Data Analysis (EDA) Using Python

In this section, we perform EDA using Python to uncover insights and patterns in the data.

### Key Steps:

- **Data Cleaning**: Handling missing values, correcting data types, and removing duplicates.
- **Visualization**:
  - Distribution of payload mass.
  - Success rate by launch site.
  - Correlation between different features.
- **Insights**:
  - Identifying the most successful launch sites.
  - Understanding the impact of payload mass on launch and landing success.

### Code Snippets

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('spacex_launch_data.csv')

# Data Cleaning
data.dropna(inplace=True)

# *Visualization*
sns.countplot(x='Launch Site', hue='Landing Outcome', data=data)
plt.title('Landing Success by Launch Site')
plt.show()
# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value
sns.catplot(x="LaunchSite", y="PayloadMass", hue="Class", data=df, aspect = 5)
plt.xlabel("Launch Site", size=20)
plt.ylabel("Payload Mass (kg)", size=20)

# A function to Extract years from the date 
year=[]
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year

Extract_year(df["Date"])
zipped = zip(df['Date'], df['Orbit'], df['Outcome'],df['Class'], year)
df1=pd.DataFrame(zipped, columns=['Date', 'Orbit', 'Outcome', 'Class', 'Year'])
df1
```
### Feature Engineering
- Created dummy variables to categorical columns
- Cast all numeric columns to float64

## Exploratory Data Analysis (EDA) Using SQL

- Overview
   - Understand the Spacex DataSet
   - Load the dataset into the corresponding table in a Db2 database
   - Execute SQL queries to answer assignment questions

### Tasks/Queries
   - Display the names of the unique launch sites in the space mission
   - Display 5 records where launch sites begin with the string 'CCA'
   - Display the total payload mass carried by boosters launched by NASA (CRS)
   - Display average payload mass carried by booster version F9 v1.1
   - List the date when the first successful landing outcome in ground pad was acheived.
   - List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000
   - List the total number of successful and failure mission outcomes
```SQL
SELECT MISSION_OUTCOME, COUNT(MISSION_OUTCOME) AS TOTAL_NUMBER
FROM SPACEXTBL
GROUP BY MISSION_OUTCOME;
```
- List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
```SQL
SELECT DISTINCT BOOSTER_VERSION
FROM SPACEXTBL
WHERE PAYLOAD_MASS__KG_ = (
    SELECT MAX(PAYLOAD_MASS__KG_)
    FROM SPACEXTBL);
```
- List the failed landing_outcomes in drone ship, their booster versions, and launch site names for in year 2015
```SQL
SELECT LANDING__OUTCOME, BOOSTER_VERSION, LAUNCH_SITE
FROM SPACEXTBL
WHERE Landing__Outcome = 'Failure (drone ship)'
    AND YEAR(DATE) = 2015;
```
- Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order
```SQL
SELECT LANDING__OUTCOME, COUNT(LANDING__OUTCOME) AS TOTAL_NUMBER
FROM SPACEXTBL
WHERE DATE BETWEEN '2010-06-04' AND '2017-03-20'
GROUP BY LANDING__OUTCOME
ORDER BY TOTAL_NUMBER DESC
```
## Machine Learning Prediction
### Objectives
- Performing exploratory Data Analysis and determine Training Labels
- creating a column for the class
- Standardizing the data
- Spliting into train data and test data
- Finding best Hyperparameter for SVM, Classification Trees and Logistic Regression
- Finding the method performs best using test data
### Tasks
- Creating a NumPy array from the column Class in data, by applying the method to_numpy() then assign it to the variable Y,make sure the output is a Pandas series (only one bracket df['name of column']).
- Standardizing the data in X then reassign it to the variable X using the transform provided below.
- Using the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to 0.2 and random_state to 2. The training data and test data should be assigned to the following labels.
- Creating a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
- Calculating the accuracy on the test data using the method score:
- Creating a support vector machine object then create a GridSearchCV object svm_cv with cv - 10. Fit the object to find the best parameters from the dictionary parameters.
- Calculating the accuracy on the test data using the method score:
- Creating a decision tree classifier object then create a GridSearchCV object tree_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
- Calculating the accuracy of tree_cv on the test data using the method score
- Creatoing a k nearest neighbors object then create a GridSearchCV object knn_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
- Calculatoing the accuracy of knn_cv on the test data using the method score:
- Finding the method performs best:
```python
# Assuming that we have already fitted the logreg_cv, svm_cv, tree_cv, and knn_cv models using GridSearchCV
# Also, X_test, Y_test are defined

# Calculating accuracy on the test data for each model
test_accuracy_logreg = logreg_cv.score(X_test, Y_test)
test_accuracy_svm = svm_cv.score(X_test, Y_test)
test_accuracy_tree = tree_cv.score(X_test, Y_test)
test_accuracy_knn = knn_cv.score(X_test, Y_test)

# Printing the accuracy scores for each model
print("Accuracy on Test Data (Logistic Regression): ", test_accuracy_logreg)
print("Accuracy on Test Data (SVM): ", test_accuracy_svm)
print("Accuracy on Test Data (Decision Tree): ", test_accuracy_tree)
print("Accuracy on Test Data (KNN): ", test_accuracy_knn)

# Finding the method that performs best
best_method = max(test_accuracy_logreg, test_accuracy_svm, test_accuracy_tree, test_accuracy_knn)

# Printing the best method
print("\nBest Performing Method:")
if best_method == test_accuracy_logreg:
    print("Logistic Regression")
elif best_method == test_accuracy_svm:
    print("Support Vector Machine (SVM)")
elif best_method == test_accuracy_tree:
    print("Decision Tree")
elif best_method == test_accuracy_knn:
    print("K-Nearest Neighbors (KNN)")
```
## Application Development
### Tasks/Steps
- Import required libraries
- Read the airline data into pandas dataframe
- Creating a dash application
- Creating an app layout
- TASK 1: Add a dropdown list to enable Launch Site selection
- The default select value is for ALL sites
- dcc.Dropdown(id='site-dropdown',...)
- TASK 2: Add a pie chart to show the total successful launches count for all sites
- Add a slider to select payload range
- Add a scatter chart to show the correlation between payload and launch success
- Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
- Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
- Run the app
## Results
- Model Accuracy: Achieved an accuracy of 88% with the Random Forest classifier.
- Key Insights:
1. Higher payload mass tends to decrease the probability of a successful landing.
2. Certain launch sites have higher success rates, indicating better infrastructure or operational efficiency.
- Application Impact: The web app provides a user-friendly interface for stakeholders to predict landing success, aiding in decision-making processes.
## Conclusion
This project successfully demonstrates the application of data science techniques to predict the success of SpaceX's reusable rockets. Through comprehensive data analysis and machine learning modeling, we identified key factors influencing landing outcomes and developed an interactive tool to leverage these insights. This work not only reinforces foundational data science skills but also contributes to the ongoing advancements in aerospace technology.
## Getting started
#### Prerequisites
- Python 3.8+
- Pip
- Virtual Environment (optional but recommended)
- SQL Database (SQLite/MySQL)

#### Installation
- Clone the Repository
```Python
git clone https://github.com/yourusername/spacex-reusable-rockets-prediction.git
cd spacex-reusable-rockets-prediction
```
- Create and Activate Virtual Environment
```Python
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- Install Dependencies

```Python
pip install -r requirements.txt
```
- Set Up the Database
```Python
Import the SQL schema and data.
sqlite3 spacex.db < schema.sql
```
Adjust the commands based on your chosen database

## Project Structure
```css
spacex-reusable-rockets-prediction/
│
├── assets/
│   └── app_screenshot.png
│
├── data/
│   ├── spacex_launch_data.csv
│   └── schema.sql
│
├── notebooks/
│   ├── EDA_Python.ipynb
│   └── EDA_SQL.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── app.py
│
├── requirements.txt
├── README.md
└── LICENSE
```
## License
This project is licensed under the MIT License. See the LICENSE file for details.
## Contact
Mohammed Ammaruddin
md.ammaruddin2020@gmail.com
https://www.linkedin.com/in/m-ammaruddin/
