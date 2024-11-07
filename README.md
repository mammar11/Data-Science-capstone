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

# Visualization
sns.countplot(x='Launch Site', hue='Landing Outcome', data=data)
plt.title('Landing Success by Launch Site')
plt.show()

