#This is Data Preprocessing for Machine Learning using Python

#Importing needed library
import numpy as np
import matplotlib.pyplot as pyl
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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
#print(x)

#Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
#print(x)

#Encoding Dependant Variable
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)

#Spliting dataset into traning set and test set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state= 1)
#print('xtrain : ', X_train)
#print('xtest : ', X_test)
#print('ytrain : ', Y_train)
#print('ytest ; ', Y_test)

#feature scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])

#result feature scaling
print('xtrain : ', X_train)
print('xtest : ', X_test)