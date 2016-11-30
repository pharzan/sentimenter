from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.utils import shuffle
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

MNB_clf = MultinomialNB()
MNB_clf.fit(train_data,targets)

print('MultinomialNB: ',MNB_clf.score(test,test_targets))

GaussianNB_clf = GaussianNB()
GaussianNB_clf.fit(train_data,targets)

print('Guassian: ',GaussianNB_clf.score(test,test_targets))

SVM_clf = SVC(kernel='linear',C=1.0)
SVM_clf.fit(train_data,targets)
print('SVM: ',SVM_clf.score(test,test_targets))

for i in test:
	Z=MNB_clf.predict([i])[0]
	color='green'
	if Z==0:
		color='red'
	elif Z==1:
		color='blue'
	plt.scatter(i[0], i[1], c=color, alpha=0.9,edgecolor="yellow")



plt.show()
