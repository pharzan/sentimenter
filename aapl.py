import quandl
import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.utils import shuffle
from matplotlib import style
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
style.use('ggplot')

CSV_File = 'WIKI-AAPL.csv'

def read_csv(CSV_File):
	df = pd.read_csv(CSV_File)
	return df


df = read_csv(CSV_File)


plt.subplot(2, 2, 1)
plt.ylim([df['Open'].min(),df['Open'].max()])
plt.plot(df.index,df['Open'],c='green')
plt.subplot(2, 2, 2)
plt.plot(df.index,df['Close'],c='red')
plt.subplot(2, 2, 3)
plt.scatter(df['Open'],df['Close'],c='blue')
plt.subplot(2, 2, 4)
plt.scatter(df.index,[df['Open']-df['Close']])


X, targets = shuffle(df['Open'],df['Close'])

train_data, test, targets, test_targets = train_test_split(df['Open'], 
	df['Close'], 
	test_size=0.40)

print(train_data.shape,targets.shape)



print(test.shape,test_targets.shape)

# test = test.reshape(3628, 1)
# test_targets = test_targets.reshape(3628, 1)
# train_data = train_data.reshape(5442,1)
# targets = targets.reshape(5442,1)
# from sklearn import linear_model
# regr = linear_model.LinearRegression()
# regr.fit(train_data, targets)
print('\x1b[0;30;41m' +' Regression: '+'\x1b[0m ',regr.score(test,test_targets))
print(regr.predict(111.60))
# train_data= np.asarray(train_data, dtype="|S6")
# targets=np.asarray(targets, dtype="|S6")

#SVM_clf = SVC(kernel='linear',C=1.0)
#SVM_clf.fit(train_data,targets)

#predicted = SVM_clf.predict(test)

#print('\x1b[6;30;42m' +' SVM: '+'\x1b[0m ',SVM_clf.score(test,test_targets))
#print(metrics.classification_report(test_targets,predicted))

#plt.show()
