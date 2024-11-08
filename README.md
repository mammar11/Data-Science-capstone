# SpaceX Reusable Rockets Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Coursera](https://img.shields.io/badge/Coursera-IBM%20Data%20Science%20Professional%20Certificate-brightgreen.svg)

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
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
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
     
- Tasks/Queries
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
   - Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order
