import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pandas.read_csv("heart.csv")

X = df[['Age','RestingBP','Cholesterol']]
Y = df['HeartDisease']

model = linear_model.LogisticRegression()

model.fit(X, Y)

age = int(input("Enter the age "))
restingBP = int(input("Enter the resting BP "))
cholestrol = int(input("Enter the Cholestrol "))

print(model.predict(np.array([[age, restingBP, cholestrol]])))