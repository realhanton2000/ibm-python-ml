import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)
abs_path = os.path.join(script_dir, 'cell_samples.csv')

cell_df = pd.read_csv(abs_path)
cell_df.head()

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
#plt.show()

#print(cell_df.dtypes)

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
#print(cell_df.dtypes)

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn import svm
#clf = svm.SVC(kernel='rbf')
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train) 

yhat = clf.predict(X_test)

from sklearn.metrics import f1_score
print('f1_score')
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print('jaccard_score')
print(jaccard_score(y_test, yhat,pos_label=2)) 