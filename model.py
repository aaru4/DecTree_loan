from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data = pd.read_csv(r"C:\Users\aarus\AppData\Local\Temp\Temp1_archive (1).zip\Default_Fin.csv")
data = data.dropna(axis=0)

y = data.Defaulted
loan_features = ['Index', 'Employed', 'Bank Balance', 'Annual Salary']
x = data[loan_features]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test)
print(acc)

