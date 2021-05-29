import os
import pandas as pd
from sklearn import preprocessing

script_dir = os.path.dirname(__file__)
abs_path = os.path.join(script_dir, 'teleCust1000t.csv')

df = pd.read_csv(abs_path)

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values
y = df['custcat'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

print('KNN')
from sklearn.neighbors import KNeighborsClassifier
k = 6
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)
yhat_proba = neigh.predict_proba(X_test)

from sklearn.metrics import f1_score
print('KNN - f1_score')
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print('KNN - jaccard_score')
print(jaccard_score(y_test, yhat, average='weighted')) 

from sklearn.metrics import log_loss
print('KNN - log_loss')
print(log_loss(y_test, yhat_proba)) 

print('Decision Trees')
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
yhat = clf.predict(X_test)
yhat_proba = clf.predict_proba(X_test)

print('DT - f1_score')
print(f1_score(y_test, yhat, average='weighted'))
print('DT - jaccard_score')
print(jaccard_score(y_test, yhat, average='weighted')) 
print('DT - log_loss')
print(log_loss(y_test, yhat_proba)) 

print('SVM')
from sklearn import svm
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)
yhat_proba = clf.predict_proba(X_test)

print('SVM - f1_score')
print(f1_score(y_test, yhat, average='weighted'))
print('SVM - jaccard_score')
print(jaccard_score(y_test, yhat, average='weighted')) 
print('SVM - log_loss')
print(log_loss(y_test, yhat_proba)) 

print('Logistic Regression')
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
yhat = clf.predict(X_test)
yhat_proba = clf.predict_proba(X_test)

print('LR - f1_score')
print(f1_score(y_test, yhat, average='weighted'))
print('LR - jaccard_score')
print(jaccard_score(y_test, yhat, average='weighted')) 
print('LR - log_loss')
print(log_loss(y_test, yhat_proba)) 
