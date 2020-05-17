# Load EDA Pkgs
import pandas as pd 
import numpy as np


# Load Data Vis Pkg
import matplotlib.pyplot as plt 
import seaborn as sns

# Load ML Pkgs
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# For Neural network (MultiLayerPerceptron)
from sklearn.neural_network import MLPClassifier


#----------------------------------------------------------------------------------------
#
#								LOADING 
#----------------------------------------------------------------------------------------


col_names = ['buying','maint','doors' ,'persons','lug_boot','safety','class']


# Load dataset
df = pd.read_csv("data/car.data",names=col_names)


#----------------------------------------------------------------------------------------
#
#								LABEL ENCODING
#----------------------------------------------------------------------------------------

# We will then label-encode our data set using either of these methods:

# Custom Function
# Label Encoder from Sklearn
# OneHot Encoding
# Pandas Get Dummies


# Custom Function
buying_label = { ni: n for n,ni in enumerate(set(df['buying']))}
maint_label = { ni: n for n,ni in enumerate(set(df['maint']))}
doors_label = { ni: n for n,ni in enumerate(set(df['doors']))}
persons_label = { ni: n for n,ni in enumerate(set(df['persons']))}
lug_boot_label = { ni: n for n,ni in enumerate(set(df['lug_boot']))}
safety_label = { ni: n for n,ni in enumerate(set(df['safety']))}
class_label = { ni: n for n,ni in enumerate(set(df['class']))}


# In our Case we will be using a custom function to help us encode our data set 
# and then map them to our values for each column respectively. 
# We will then save these labels as dictionaries and use it for building the options sections of our ML app.


print('Custom encoding or label encoding?')
choice = int(input("Custom = 1 Label = 2 \n"))

if choice == 1:
	df['buying'] = df['buying'].map(buying_label)
	df['maint'] = df['maint'].map(maint_label)
	df['doors'] = df['doors'].map(doors_label)
	df['persons'] = df['persons'].map(persons_label)
	df['lug_boot'] = df['lug_boot'].map(lug_boot_label)
	df['safety'] = df['safety'].map(safety_label)
	df['class'] = df['class'].map(class_label)
elif choice ==2:
	from sklearn.preprocessing import LabelEncoder
	lb=LabelEncoder()
	for i in df.columns:
		df[i]=lb.fit_transform(df[i])
else:
	print("You didnt choose correctly")
	exit()


#----------------------------------------------------------------------------------------
#
#								BUILDING MODEL
#----------------------------------------------------------------------------------------

# To summarize we will be using 3 different ML algorithms 
#(LogisticRegression,Naive Bayes and Multi-Layer Perceptron Classifier).

# We will first split our dataset into training and test dataset.


Xfeatures = df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
ylabels = df['class']

X_train, X_test, Y_train, Y_test = train_test_split(Xfeatures, ylabels, test_size=0.30, random_state=7)


# LOGISTICAL REGRESSION
# Using - Logisitic Regression
print('Running Logistical Regression')
logit = LogisticRegression()
logit.fit(X_train, Y_train)

print("Accuracy Score:",accuracy_score(Y_test, logit.predict(X_test)))
print('')

print('Running Neural Network')
# Using Neural Network
nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
nn_clf.fit(X_train,Y_train)

print("Accuracy Score:",accuracy_score(Y_test, nn_clf.predict(X_test)))


#----------------------------------------------------------------------------------------
#
#								SAVING MODEL
#----------------------------------------------------------------------------------------

import joblib
print('Saving logistical model...')
logit_model = open("logit_car_model.pkl","wb")
joblib.dump(logit,logit_model)
logit_model.close()
print('Saved')

print('Saving Neural Network Model...')
nn_clf_model = open("nn_clf_car_model.pkl","wb")
joblib.dump(nn_clf,nn_clf_model)
nn_clf_model.close()
print('Saved')
print('Job complete')

