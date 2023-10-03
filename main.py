import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

df.head(10)

df.info()

df.describe()

df.shape

df.dropna(inplace = True)

df['id'].value_counts()
df = df.drop('id',axis = 1)


df = df.join(pd.get_dummies(df['work_type'])).drop('work_type',axis =1)
df = df.join(pd.get_dummies(df['Residence_type'])).drop('Residence_type',axis =1)
df = df.join(pd.get_dummies(df['smoking_status'])).drop('smoking_status',axis =1)
df = df.join(pd.get_dummies(df['gender'])).drop('gender',axis =1)


labeling = LabelEncoder()
df['ever_married'] = labeling.fit_transform(df['ever_married'])


model = RandomForestClassifier()

x, y = df.drop('stroke',axis = 1), df['stroke']
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
model.fit(x_train, y_train)
model.score(x_test,y_test)