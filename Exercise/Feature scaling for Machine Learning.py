# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the Wine Quality Red dataset
df = pd.read_csv('winequality-red.csv', delimiter = ';')

# Separate features and target
X = df.drop('quality', axis = 1)
Y = df['quality']


# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state= 42)

# Create an instance of the StandardScaler class
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply the transform to the test set


# Print the scaled training and test datasets
print('X_train : ', X_train)
print('X_test : ', X_test)
