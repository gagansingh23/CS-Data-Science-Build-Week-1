
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.spatial
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

from knn import KNN

path = os.path.join(os.getcwd(),"data/data.csv")
df = pd.read_csv(path)

target = df['Grade']
features = df.drop('Grade', axis=1)

target = np.array(target)
features = np.array(features)

X, y = features, target

#Split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)

#run the test
clf = KNN(k=3)
clf.fit(X_train, y_train)
y_pred = clf.predict([[1500, 3.5]])
print(y_pred)

#train and test
clf = KNN(k=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("custom KNN classification accuracy", accuracy_score(y_test, y_pred))

