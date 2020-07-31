from sklearn.neighbors import KNeighborsClassifier
import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = os.path.join(os.getcwd(),"data/data.csv")
df = pd.read_csv(path)

target = df['Grade']
features = df.drop('Grade', axis=1)

target = np.array(target)
features = np.array(features)

X, y = features, target

#Split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)


classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict([[1700, 3.5]])
print(y_pred)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("sklearn model classification accuracy", accuracy_score(y_test, y_pred))

