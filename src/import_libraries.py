# ==========================================
# File 1: import_libraries.py
# Purpose: Import all required libraries and explain their purpose in the project.
# ==========================================

# 1. Pandas - Used for data manipulation, cleaning, and analysis.
# It provides DataFrame structures to easily handle tabular data (CSV).
import pandas as pd

# 2. NumPy - Used for numerical operations on arrays.
# It helps in handling math operations and high-performance array manipulations.
import numpy as np

# 3. Matplotlib (pyplot) - Used for data visualization.
# It allows us to generate static graphs like histograms, bar charts, and scatter plots.
import matplotlib.pyplot as plt

# 4. Scikit-learn (sklearn) - The main machine learning library in Python.
# We import several modules from it:
# - train_test_split: to divide our dataset into training and testing subsets.
# - DecisionTreeClassifier & RandomForestClassifier: algorithms for training our model.
# - GridSearchCV: to perform hyperparameter tuning to get the best model performance.
# - accuracy_score, classification_report, confusion_matrix: to evaluate model performance.
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 5. Joblib - Used for saving and loading Python objects efficiently.
# In this project, it is used to save the trained model (.pkl file) and load it for predictions.
import joblib

# 6. Streamlit - Used to build interactive, user-friendly web applications.
# It enables us to create the front-end user interface where users input soil data to get crop recommendations.
import streamlit as st

print("Libraries imported successfully!")
