# Data Classification

**Author:** Omar A.Sabri

## Overview

This project focuses on classifying student performance based on various features. The classification is achieved through various machine learning techniques using Python libraries.

## Table of Contents

1. [Import Libraries](#import-libraries)
2. [Reading Data](#reading-data)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Selection](#feature-selection)
6. [Model Building](#model-building)
7. [Model Evaluation](#model-evaluation)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Conclusion](#conclusion)
10. [How to Run](#how-to-run)
11. [License](#license)

## 1. Import Libraries

In this section, we import the necessary libraries for data manipulation, visualization, and machine learning.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')
