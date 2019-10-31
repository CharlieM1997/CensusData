# -*- coding: utf-8 -*-
"""
Fundamentals of Machine Learning
Classifier for Census Data

@author: 164776
"""

from astropy.io import ascii
import pandas as pd
import numpy as np
import missingno as msno
from sklearn_pandas import CategoricalImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

#First step is to load the training and test data into the script, using Astropy.
filename = 'adult.data.txt'
testname = 'adult.test.txt'
#The column names are put into a list so that they can be added.
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-region', 
        '<=50K, >50K']
training_data = ascii.read(filename, names=names, data_start=4)
#training_data.show_in_browser()
test_data = ascii.read(testname, names=names, data_start=4)
#test_data.show_in_browser()

#Data Preprocessing

#Both the training and test data sets will have some categories be collapsed.
#This is to get rid of high variance in the datasets.
#Question marks are also changed to missing values.
training_data['workclass'] = pd.Series(training_data['workclass']).replace({
        'Self-emp-not-inc': 'Self-Emp-Not-Inc', 'State-gov': 'State',
        'Federal-gov': 'Federal', 'Local-gov': 'Local', '?': None,
        'Self-emp-inc': 'Self-Emp-Inc', 'Without-pay': 'Unpaid',
        'Never-worked': 'Never-Worked'})
training_data['education'] = pd.Series(training_data['education']).replace({
        '1st-4th': 'Primary', '5th-6th': 'Primary', '7th-8th': 'Junior-High',
        '9th': 'High-School', 'Assoc-acdm': 'High-School',
        'Assoc-voc': 'High-School', '10th': 'High-School', 'HS-grad': 'HS-Grad',
        '11th': 'HS-Grad', '12th': 'HS-Grad', 'Bachelors': 'Under-Grad',
        'Some-college': 'Under-Grad'})
training_data['marital-status'] = pd.Series(training_data['marital-status']).replace({
        'Divorced': 'Separated', 'Married-civ-spouse': 'Married',
        'Married-AF-spouse': 'Married', 'Never-married': 'Single',
        'Married-spouse-absent': 'Spouse-Absent'})
training_data['occupation'] = pd.Series(training_data['occupation']).replace({
        '?': None, 'Adm-clerical': 'Clerical', 'Craft-repair':
        'Low-Skill-Labour', 'Handlers-cleaners': 'Low-Skill-Labour',
        'Machine-op-inspect': 'Low-Skill-Labour', 'Other-service':
        'Low-Skill-Labour', 'Priv-house-serv': 'Low-Skill-Labour',
        'Protective-serv': 'Low-Skill-Labour', 'Prof-speciality':
        'High-Skill-Labour', 'Sales': 'High-Skill-Labour', 'Tech-support':
        'High-Skill-Labour', 'Transport-moving': 'High-Skill-Labour',
        'Armed-Forces': 'Armed-Forces', 'Farming-fishing': 'Agriculture',
        'Exec-managerial': 'Executive'})
training_data['relationship'] = pd.Series(training_data['relationship']).replace({
        'Not-in-family': 'Sole-Resident', 'Other-relative': 'With-Relative'})
training_data['native-region'] = pd.Series(training_data['native-region']).replace({
        '?': None, 'South': 'SE-Asia', 'Vietnam': 'SE-Asia', 'Laos': 'SE-Asia',
        'Cambodia': 'SE-Asia', 'Thailand': 'SEAsia', 'China': 'Asia',
        'HongKong': 'Asia', 'Taiwan': 'Asia', 'India': 'Asia', 'Philippines': 'Asia',
        'Iran': 'Asia', 'Japan': 'Asia', 'England': 'West-EU', 'Scotland': 'West-EU',
        'Cuba': 'Central-America', 'Dominican-Republic': 'Central-America',
        'Guatemala': 'Central-America', 'Haiti': 'Central-America',
        'Honduras': 'Central-America', 'Jamaica': 'Central-America',
        'Mexico': 'Central-America', 'Nicaragua': 'Central-America',
        'Puerto-Rico': 'Central-America', 'El-Salvador': 'Central-America',
        'Ecuador': 'South-America', 'Peru': 'South-America', 'Columbia':
        'South-America', 'Trinada&Tobago': 'South-America', 'France': 'West-EU',
        'Germany': 'West-EU', 'Greece': 'West-EU', 'Holand-Netherlands': 'West-EU',
        'Italy': 'West-EU', 'Ireland': 'West-EU', 'Portugal': 'West-EU',
        'Hungary': 'East-EU', 'Poland': 'East-EU', 'Yugoslavia': 'East-EU',
        'Outlying-US(Guam-USVI-etc)': 'Oceania'})
    
test_data['workclass'] = pd.Series(test_data['workclass']).replace({
        'Self-emp-not-inc': 'Self-Emp-Not-Inc', 'State-gov': 'State',
        'Federal-gov': 'Federal', 'Local-gov': 'Local', '?': None,
        'Self-emp-inc': 'Self-Emp-Inc', 'Without-pay': 'Unpaid',
        'Never-worked': 'Never-Worked'})
test_data['education'] = pd.Series(test_data['education']).replace({
        '1st-4th': 'Primary', '5th-6th': 'Primary', '7th-8th': 'Junior-High',
        '9th': 'High-School', 'Assoc-acdm': 'High-School',
        'Assoc-voc': 'High-School', '10th': 'High-School', 'HS-grad': 'HS-Grad',
        '11th': 'HS-Grad', '12th': 'HS-Grad', 'Bachelors': 'Under-Grad',
        'Some-college': 'Under-Grad'})
test_data['marital-status'] = pd.Series(test_data['marital-status']).replace({
        'Divorced': 'Separated', 'Married-civ-spouse': 'Married',
        'Married-AF-spouse': 'Married', 'Never-married': 'Single',
        'Married-spouse-absent': 'Spouse-Absent'})
test_data['occupation'] = pd.Series(test_data['occupation']).replace({
        '?': None, 'Adm-clerical': 'Clerical', 'Craft-repair':
        'Low-Skill-Labour', 'Handlers-cleaners': 'Low-Skill-Labour',
        'Machine-op-inspect': 'Low-Skill-Labour', 'Other-service':
        'Low-Skill-Labour', 'Priv-house-serv': 'Low-Skill-Labour',
        'Protective-serv': 'Low-Skill-Labour', 'Prof-speciality':
        'High-Skill-Labour', 'Sales': 'High-Skill-Labour', 'Tech-support':
        'High-Skill-Labour', 'Transport-moving': 'High-Skill-Labour',
        'Armed-Forces': 'Armed-Forces', 'Farming-fishing': 'Agriculture',
        'Exec-managerial': 'Executive'})
test_data['relationship'] = pd.Series(test_data['relationship']).replace({
        'Not-in-family': 'Sole-Resident', 'Other-relative': 'With-Relative'})
test_data['native-region'] = pd.Series(test_data['native-region']).replace({
        '?': None, 'South': 'SE-Asia', 'Vietnam': 'SE-Asia', 'Laos': 'SE-Asia',
        'Cambodia': 'SE-Asia', 'Thailand': 'SEAsia', 'China': 'Asia',
        'HongKong': 'Asia', 'Taiwan': 'Asia', 'India': 'Asia', 'Philippines': 'Asia',
        'Iran': 'Asia', 'Japan': 'Asia', 'England': 'West-EU', 'Scotland': 'West-EU',
        'Cuba': 'Central-America', 'Dominican-Republic': 'Central-America',
        'Guatemala': 'Central-America', 'Haiti': 'Central-America',
        'Honduras': 'Central-America', 'Jamaica': 'Central-America',
        'Mexico': 'Central-America', 'Nicaragua': 'Central-America',
        'Puerto-Rico': 'Central-America', 'El-Salvador': 'Central-America',
        'Ecuador': 'South-America', 'Peru': 'South-America', 'Columbia':
        'South-America', 'Trinada&Tobago': 'South-America', 'France': 'West-EU',
        'Germany': 'West-EU', 'Greece': 'West-EU', 'Holand-Netherlands': 'West-EU',
        'Italy': 'West-EU', 'Ireland': 'West-EU', 'Portugal': 'West-EU',
        'Hungary': 'East-EU', 'Poland': 'East-EU', 'Yugoslavia': 'East-EU',
        'Outlying-US(Guam-USVI-etc)': 'Oceania'})

#The astropy tables are converted into pandas dataframes, as this is easier to
#work and plot with.    
df = training_data.to_pandas()
dftest = test_data.to_pandas()

#This code generates a stacked graph to see how much data is missing in the
#training data.
counts = df.count()
nulls = df.isnull().sum()
pltdf = pd.DataFrame({'Known data': counts, 'Missing data': nulls},
    index=training_data.colnames)
ax = pltdf.plot.bar(color=['b', 'r'], stacked=True)
ax.legend(loc=4)
ax.set_xlabel('Missing training data')
ax.set_ylim((25000, 33000))

#This creates a dendrogram to check for correlations in the missing data.
#If there are correlations, it suggests that the missing data is MNAR.
msno.dendrogram(df)

#The rows with missing data are dropped from the training and test data.
dfdropped = df.dropna(axis=0)
dftestdropped = dftest.dropna(axis=0)

#A categorical imputer is used to replace the missing values with the most
#frequent values in each column.
ci = CategoricalImputer(strategy='mode')
df['workclass'] = ci.fit_transform(np.asarray(df['workclass']))
df['occupation'] = ci.fit_transform(np.asarray(df['occupation']))
df['native-region'] = ci.fit_transform(np.asarray(df['native-region']))
dftest['workclass'] = ci.fit_transform(np.asarray(dftest['workclass']))
dftest['occupation'] = ci.fit_transform(np.asarray(dftest['occupation']))
dftest['native-region'] = ci.fit_transform(np.asarray(dftest['native-region']))

#Both the removed and imputed dataframes are put under one hot encoding.
#This is done as scikit-learn classifiers only work with numerical data.
dfonehot = pd.get_dummies(df)
dfdroppedonehot = pd.get_dummies(dfdropped)
dftestonehot = pd.get_dummies(dftest)
dftestdroppedonehot = pd.get_dummies(dftestdropped)

#The training and test data is further split into training and validation sets,
#ready to be fit onto the classifiers.
X_train, y_train = dfonehot.iloc[:, :-1], dfonehot.iloc[:, -1]
Xdropped_train, ydropped_train = dfdroppedonehot.iloc[:, :-1], dfdroppedonehot.iloc[:, -1]
X_test, y_test = dftestonehot.iloc[:, :-1], dftestonehot.iloc[:, -1]
Xdropped_test, ydropped_test = dftestdroppedonehot.iloc[:, :-1], dftestdroppedonehot.iloc[:, -1]

#The first model to be used is a logistic regression model. The multi class
#feature is set to multinomial, due to the high dimensionality of the data.
#For each model, the model is fit to the training data, and is used to predict
#the test data and finally scores the accuracy.
lrmodel = LogisticRegression(fit_intercept=True, random_state=164776,
                             solver='lbfgs', multi_class='multinomial')
lrmodel.fit(X_train, y_train)
lrpred = lrmodel.predict(X_test)
print('Accuracy of logistic regression classifier on test set w/ imputed values: {:.3f}'
      .format(lrmodel.score(X_test, y_test)))

#Confusion matrices are also created to measure the sensitivity and specificity.
#This is because accuracy alone is not enough to decide how effective a model
#is when testing against new data.
results = confusion_matrix(y_test, lrpred)
total=(sum(sum(results)))
print('Sensitivity of logistic regression classifier w/ imputed values: {:.3f}'
      .format(results[0,0]/(results[0,0]+results[0,1])))
print('Specificity of logistic regression classifier w/ imputed values: {:.3f}'
      .format(results[1,1]/(results[1,0]+results[1,1])))

#For the logistic regression models, an ROC Curve is also generated from its
#results to see how good the classifier truly is. A good classifier will stay
#in the upper left corner, and a random classifier will follow the dotted line.
logit_roc_auc = roc_auc_score(y_test, lrpred)
fpr, tpr, thresholds = roc_curve(
        y_test, lrmodel.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')

#The same model is also used to test against the data where the missing
#values were removed.
lrmodel1 = LogisticRegression(fit_intercept=True, random_state=164776,
                             solver='lbfgs', multi_class='multinomial')
lrmodel1.fit(Xdropped_train, ydropped_train)
lrpred1 = lrmodel1.predict(Xdropped_test)
print('Accuracy of logistic regression classifier on test set w/o missing values: {:.3f}'
      .format(lrmodel1.score(Xdropped_test, ydropped_test)))

results = confusion_matrix(ydropped_test, lrpred1)
total=(sum(sum(results)))
print('Sensitivity of logistic regression classifier w/o missing values: {:.3f}'
      .format(results[0,0]/(results[0,0]+results[0,1])))
print('Specificity of logistic regression classifier w/o missing values: {:.3f}'
      .format(results[1,1]/(results[1,0]+results[1,1])))

#For the logistic regression models, an ROC Curve is also generated from its
#results to see how good the classifier truly is. A good classifier will stay
#in the upper left corner, and a random classifier will follow the dotted line.
logit_roc_auc = roc_auc_score(ydropped_test, lrpred1)
fpr, tpr, thresholds = roc_curve(
        ydropped_test, lrmodel1.predict_proba(Xdropped_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')

#A Linear SVM Classifier is also used to classify the data, and at the end
#it will be compared to the logistic regression classifier.
lsvcmodel = LinearSVC(random_state=164776)
lsvcmodel.fit(X_train, y_train)
lsvcpred = lsvcmodel.predict(X_test)
print('Accuracy of linear SVM classifier on test set w/ imputed values: {:.3f}'
      .format(lsvcmodel.score(X_test, y_test)))

#A confusion matrix is also generated for the Linear SVM Classifier.
results = confusion_matrix(y_test, lsvcpred)
total=(sum(sum(results)))
print('Sensitivity of linear SVM classifier w/ imputed values: {:.3f}'
      .format(results[0,0]/(results[0,0]+results[0,1])))
print('Specificity of linear SVM classifier w/ imputed values: {:.3f}'
      .format(results[1,1]/(results[1,0]+results[1,1])))

#The Linear SVM Classifier is also used to classify the data when the missing
#values are removed. Because the accuracy is lower in this instance, there's
#no need to generate a confusion matrix or ROC Curve.
lsvcmodel1 = LinearSVC(random_state=164776)
lsvcmodel1.fit(Xdropped_train, ydropped_train)
lsvcpred1 = lsvcmodel1.predict(Xdropped_test)
print('Accuracy of linear SVM classifier on test set w/o missing values: {:.3f}'
      .format(lsvcmodel1.score(Xdropped_test, ydropped_test)))

#The final accuracy results are compared on a horizontal bar chart.
dfresults = pd.DataFrame({'Classifier':
    ['Logistic Regression Classifier\nw/ imputed values',
     'Logistic Regression Classifier \nw/o missing values',
     'Linear SVM Classifier \nw/ imputed values',
     'Linear SVM Classifier \nw/ imputed values'],
     'Accuracy': [0.798, 0.878, 0.796, 0.785]})
ax = dfresults.plot.barh(x='Classifier', y='Accuracy')
ax.set_title('Classifier Accuracy')
ax.set_xlim([0.0, 1.0])
ax.set_ylabel('')
for p in ax.patches:
    ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()),
                xytext=(5, 10), textcoords='offset points')