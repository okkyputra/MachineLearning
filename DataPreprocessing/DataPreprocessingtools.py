#This is Data Preprocessing for Machine Learning using Python

#Importing needed library
import numpy as np
import matplotlib.pyplot as pyl
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#Importing needed dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#print(x)
#print(y)

#Handle missing data on dataset
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)

#Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

#Encoding Dependant Variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#Spliting dataset into traning set and test set
