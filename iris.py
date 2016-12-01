from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC, NuSVC
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
data = load_iris()

for i in range(0,len(data.data)-1):
	x, y = [data.data[i][0], data.data[i][1]]
	color = 'green'
	if(data.target[i]==0):
		color='red'
	elif(data.target[i]==1):
		color='blue'
	plt.scatter(x, y, s=100, c=color, alpha=1, edgecolor="black")

print(x,y)
plt.scatter(x,y)
data.data , data.target = shuffle(data.data, data.target)

train_data = data.data[100:]
test = data.data[:50]

targets = data.target[100:]
test_targets = data.target[:50]
print_headers=['','','','']
# print(tabulate(train_data[10:],headers=print_headers, tablefmt='orgtbl'))
# print(tabulate(targets[10:],headers=print_headers, tablefmt='orgtbl'))
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(train_data, targets)

print('\x1b[0;30;41m' +' Regression: '+'\x1b[0m ',regr.score(test,test_targets))

MNB_clf = MultinomialNB()
MNB_clf.fit(train_data,targets)
predicted = MNB_clf.predict(test)
#print('MultinomialNB Report: ',metrics.classification_report(test_targets,predicted))
print('\x1b[6;30;42m' +' MultinomialNB: '+'\x1b[0m ',MNB_clf.score(test,test_targets))

GaussianNB_clf = GaussianNB()
GaussianNB_clf.fit(train_data,targets)

predicted = GaussianNB_clf.predict(test)

print('\x1b[6;30;42m' +' Guassian: '+'\x1b[0m ',GaussianNB_clf.score(test,test_targets))
print(metrics.classification_report(test_targets,predicted))

print(type(train_data),train_data)
print(type(targets),targets)
SVM_clf = SVC(kernel='linear',C=1.0)
SVM_clf.fit(train_data,targets)

predicted = SVM_clf.predict(test)

print('\x1b[6;30;42m' +' SVM: '+'\x1b[0m ',SVM_clf.score(test,test_targets))
print(metrics.classification_report(test_targets,predicted))

SVC_radial = SVC(kernel='rbf')
SVC_radial.fit(train_data,targets)
predicted = SVC_radial.predict(test)

print('\x1b[6;30;42m' +' SVM Radial '+'\x1b[0m ',SVC_radial.score(test,test_targets))
print(metrics.classification_report(test_targets,predicted))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_data, targets) 
predicted = knn.predict(test)

print('\x1b[6;30;42m' +' KNN: '+'\x1b[0m ',knn.score(test,test_targets))
print(metrics.classification_report(test_targets,predicted))


from sklearn.tree import DecisionTreeClassifier
tree_CLF = DecisionTreeClassifier()
tree_CLF.fit(train_data, targets)
predicted = tree_CLF.predict(test)

print('\x1b[6;30;42m' +' Tree: '+'\x1b[0m ', tree_CLF.score(test,test_targets))
print(metrics.classification_report(test_targets,predicted))
