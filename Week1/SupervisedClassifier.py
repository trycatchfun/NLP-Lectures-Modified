import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# importing the dataset
dataset = pd.read_csv('Iris.csv')
# print(dataset.head())

# drop the id column
dataset = dataset.drop('Id', axis=1)
# print(dataset.head())

# Summary of dataset
# print(dataset.shape)

# more information
# print(dataset.info())
# print(dataset.describe())
# print(dataset.groupby('Species').size())

# Visualization
# Box plot or whisker plot
d