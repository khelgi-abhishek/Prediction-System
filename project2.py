# -*- coding: utf-8 -*-
"""project2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w_00QTFRFC6pEz-qSVoXBwr2PYAbpT8p
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore")

diabetes_dataset=pd.read_csv('/content/sample_data/diabetes.csv')

pd.read_csv

diabetes_dataset.head()

diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

print(X)

print(Y)

scaler=StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)

X=standardized_data
Y=diabetes_dataset['Outcome']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction , Y_train)

print('Accuracy score of the training dat :',training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction , Y_test)

print('Accuracy score of the test data :',test_data_accuracy)

input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
standardized_data=scaler.transform(input_data_reshaped)
print(standardized_data)
prediction = classifier.predict(standardized_data)
print(prediction)
if(prediction[0]==0):
  print('The person is not diabetic')
else:
    print('The person is diabetic')

import pickle

filename='trained_model.sav'
pickle.dump(classifier,open(filename,'wb'))

loaded_model=pickle.load(open('trained_model.sav','rb'))

input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
standardized_data=scaler.transform(input_data_reshaped)
print(standardized_data)
prediction = loaded_model.predict(standardized_data)
print(prediction)
if(prediction[0]==0):
  print('The person is not diabetic')
else:
    print('The person is diabetic')