## Project 1 ##
## Oranges Vs Grapefruits ##
## By Brandon Deliz ##

## Aim of this project is to showcase how pandas and numpy is used to ##
## showcase data and provide analyzation results involving designated ##
## precision rates, recall rates, and much more data analysis in general ##

## I also used other methods not mentioned in the class such as Logistic ##
## Regression, Linear Regression, and even the usage of the KNN algorithm ##

## Confusion Matrix is also used as well for the project ##
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
from sklearn.model_selection import GridSearchCV

## First line of code sets up the csv file that would be read via the panda module ##

citrus_ds = pd.read_csv('C:/Users/Owner/OneDrive/citrus.csv')
print(citrus_ds) ## Print statement
## Citrus_ds is uploaded to numpy array for usage with algorithms ##

name = citrus_ds['name'].to_numpy()

## Creates Y for table ##

y = np.zeros(len(name))
y [name == "grapefruit"] = 1

## Creates X for table ##

X = np.array([citrus_ds['diameter'].to_numpy(), citrus_ds['weight'].to_numpy(), citrus_ds['red'].to_numpy(),
             citrus_ds['green'].to_numpy(), citrus_ds['blue'].to_numpy()])

X = np.transpose(X)
## Splits Training and Test sets of data ##

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

## Fruit Class Lables ##
## Oranges = [1], Grapefruits = [0] ##

## The total number of samples ##

print(len(X))
print(X.shape[0])

## Number of Oranges ##

print("Oranges:", len(X[y == 1]))
print("Oranges:", X[y == 1].shape[0])

# Number of GrapeFruits ##

print("Grapefruits:", len(X[y == 0]))
print("Grapefruits:", X[y == 0].shape[0])

## Print out the training and test sets ##

print('training set', X_train.shape)
print('test set', X_test.shape)

## Creation and training of Regression Model ##

logitR = LogisticRegression(C = 1, penalty = 'l1', solver = 'saga', random_state = 0, max_iter=10000)
logitR.fit(X_train, y_train)


## Evaluation to predict class labels  ##

logitR.predict(X_train[:5])

## True classes ##

y_train[0:5]

## Probability output ##

logitR.predict_proba(X_train[:5])


## Model Performance ##

acc_train = logitR.score(X_train, y_train)
acc_test = logitR.score(X_test, y_test)
print('training acc:', acc_train, 'test acc:', acc_test)

# Calculate precision and Accuracy using the cfm(Confusion Matrix) ##
y_test_pred = logitR.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

## Recall rate for Oranges ##

print("recall", cm[0][0]/cm[0][0] + cm[0][1])

# Recall rate for GrapeFruits ##


print("precision", cm[0][0]/cm[0][0] + cm[1][0])

## Parameter searching using GridSearchCV ##

rs = 0
logitR = LogisticRegression(solver="liblinear",random_state=0, max_iter=10000)

param_grid = {
            'penalty' : ['l2','l1'],  
            'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

clf = GridSearchCV(estimator=logitR, cv=3, param_grid=param_grid , scoring='accuracy', verbose=3)
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

####################################################################################

## Evaluation metrics ##

## Test Accuracy ##
acc_test = clf.score(X_test, y_test)
print(acc_test) 
y_test_pred = clf.predict(X_test)

## Confusion Matrix ##

cm = confusion_matrix(y_test, y_test_pred)
print(cm)

## Precision Rate ##
print("precision", cm[0][0]/(cm[0][0] + cm[1][0]))

#Recall Rate ##
print("recall", cm[0][0]/(cm[0][0] + cm[0][1]))


#####################################################################################

## Linear Regression Model Code Portion ##

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

## Citrus_ds table shown visually for data depiction of original table ##

citrus_ds = pd.read_csv('C:/Users/Owner/OneDrive/citrus.csv')
citrus_ds

citrus_ds.shape
## Info displays datatype, and Non-Null count of designated columns ##
citrus_ds.info()
## Describes variables such as count, mean, etc for dataset ##
citrus_ds.describe()

## Evaluation metrics ##

## Test Accuracy ##
acc_test = clf.score(X_test, y_test)
print(acc_test) 
y_test_pred = clf.predict(X_test)

## Confusion Matrix ##

cm = confusion_matrix(y_test, y_test_pred)
print(cm)

## Precision Rate ##
print("precision", cm[0][0]/(cm[0][0] + cm[1][0]))

#Recall Rate ##
print("recall", cm[0][0]/(cm[0][0] + cm[0][1]))

X_train
y_train

###################################################################################


# KNN(X_train, y_train, X_test, k=7)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)

neigh.fit(X_train, y_train)
predictions = neigh.predict(X_test)
print(predictions)
# (y_test[:100] == predictions).sum()/len(predictions)
cm = confusion_matrix(y_test, predictions)
print(cm)

acc_test = clf.score(X_test, y_test)
print(acc_test)
y_test_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print(cm)

print("precision", cm[0][0]/(cm[0][0] + cm[1][0]))

print("recall", cm[0][0]/(cm[0][0] + cm[0][1]))


## Best values for KNN Algorithm ##
from sklearn.model_selection import cross_val_score


for k in [1,2,3,4,5,6,7,8,9,10]:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X_test, y_test, cv = 5, scoring = 'accuracy')
    scores = cross_val_score(knn, X_test, y_test, cv=5, scoring = 'precision_macro')
    #print(scores)
    print("k: %d. Precision: %0.2f (+/- %0.2f)" % (k, scores.mean(), scores.std()))

##############################################################################################




