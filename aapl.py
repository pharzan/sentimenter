import quandl
import datetime, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.utils import shuffle
from matplotlib import style
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

style.use('ggplot')
CSV_File = 'WIKI-AAPL.csv'

def read_csv(CSV_File):
	df = pd.read_csv(CSV_File)
	return df


df = read_csv(CSV_File)


plt.subplot(4, 2, 1)
plt.ylim([df['Open'].min(),df['Open'].max()])
plt.plot(df.index,df['Open'],c='green')
plt.subplot(4, 2, 2)
plt.plot(df.index,df['Close'],c='red')
plt.subplot(4, 2, 3)
plt.scatter(df['Open'],df['Close'],c='blue')
plt.subplot(4, 2, 4)
plt.scatter(df.index,[df['Open']-df['Close']])
plt.subplot(4, 2, 5)
plt.plot(df.index,df['Close'],c='red')
plt.subplot(4, 2, 6)
plt.plot(df.index,df['Close'],c='red')
plt.subplot(4, 2, 7)
plt.plot(df.index,df['Close'],c='red')


df = shuffle(df)

train_data, test, targets, test_targets = df['Open'][:500],df['Open'][500:1000], df['Close'][:500], df['Close'][500:1000]
	

print('Train len>>',train_data.shape,'Test Len>>',test.shape, 'df SHAPE:', df.shape)

train_data = np.array(train_data).reshape(len(train_data),1)
targets = np.array(targets)

test = np.array(test).reshape(len(test),1)
test_targets = np.array(test_targets)

from sklearn import linear_model
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(train_data, targets)
print('LINEAR COEFF: ',regr.coef_,'INTERCEPT:',regr.intercept_)

predicted = regr.predict(train_data)

print('YESTERDAY:',regr.predict(111.60))
print('\x1b[0;30;41m' +' Regression Score: '+'\x1b[0m ',regr.score(test,test_targets))

print('\x1b[0;30;41m' +' MEAN ABS_ERROR: '+'\x1b[0m ',mean_absolute_error(test_targets, predicted))

err = targets - predicted
err = pd.DataFrame(err)
print(err.describe())
print('\x1b[0;30;41m' +' R^2: '+'\x1b[0m ',r2_score(test_targets, predicted))

# print(regr.predict(111.60))

SVR_regr_lin = SVR(kernel='linear')
SVR_regr_pol = SVR(kernel='poly', degree=2)
SVR_regr_rbf = SVR(kernel='rbf', C=10e3 ,gamma=0.1)


SVR_regr_lin.fit(train_data, targets)
print('\x1b[0;30;41m' +' SVR: '+'\x1b[0m ',SVR_regr_lin.score(test,test_targets))

# SVR_regr_pol.fit(train_data, targets)
# print('\x1b[0;30;41m' +' Poly: '+'\x1b[0m ',SVR_regr_pol.score(test,test_targets))

SVR_regr_rbf.fit(train_data, targets)
print('\x1b[0;30;41m' +' RBF: '+'\x1b[0m ',SVR_regr_rbf.score(test,test_targets))

print(cross_val_score(regr, test, test_targets))

df['reg_Predicted'] = regr.predict(df['Open'].values.reshape(-1,1))

df['SVR_lin_Predicted'] = SVR_regr_lin.predict(df['Open'].values.reshape(-1,1))
df['SVR_rbd_Predicted'] = SVR_regr_rbf.predict(df['Open'].values.reshape(-1,1))

plt.subplot(4, 2, 5)
plt.scatter(df.index,df['reg_Predicted'],c='green', label='LINEAR REGRESSION')

plt.subplot(4, 2, 6)
plt.scatter(df.index,df['SVR_lin_Predicted'],c='green', label='SVR LINEAR MODEL')

plt.subplot(4, 2, 7)
plt.scatter(df.index,df['SVR_rbd_Predicted'],c='green', label='RBF MODEL')

#print('\x1b[6;30;42m' +' SVM: '+'\x1b[0m ',SVM_clf.score(test,test_targets))
#print(metrics.classification_report(test_targets,predicted))

plt.show()
