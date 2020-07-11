# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('Folds5x2_pp.csv')
indep_feat = dataset.iloc[:, :-1].values
dep_feat = dataset.iloc[:,4].values

#Splitting the Dataset into the Training set and Testing Set
from sklearn.model_selection import train_test_split
indep_feat_train, indep_feat_test, dep_feat_train, dep_feat_test= train_test_split(indep_feat, dep_feat, test_size =0.1, random_state=0)


#Fitting Random Forest Regression to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=50, random_state =0 )
regressor.fit(indep_feat_train,dep_feat_train)

#Predicting a new result
dep_feat_predRFR = regressor.predict(indep_feat_test)

#Visualising the Regression Results
plt.scatter(dep_feat_test,dep_feat_predRFR, s=1 )
plt.xlabel('True Values')
plt.ylabel('Predictions')

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(dep_feat_test, dep_feat_predRFR)))
print('R2 score:' , metrics.r2_score(dep_feat_test, dep_feat_predRFR))
print('Mean Squared Error:', metrics.mean_squared_error(dep_feat_test, dep_feat_predRFR))
print('Mean Absolute Error: ', metrics.mean_absolute_error( dep_feat_test, dep_feat_predRFR))

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(indep_feat_train, dep_feat_train)

dep_feat_predSVR = svr_reg.predict(indep_feat_test)

plt.scatter(dep_feat_test,dep_feat_predSVR, s=1) 
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.show()

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(dep_feat_test, dep_feat_predSVR)))
print('R2:', metrics.r2_score(dep_feat_test, dep_feat_predSVR))
print('Mean Absolute Error: ', metrics.mean_absolute_error(dep_feat_test, dep_feat_predSVR))


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(indep_feat_train, dep_feat_train)

dep_feat_predMLR = lin_reg.predict(indep_feat_test)

plt.scatter(dep_feat_test,dep_feat_predMLR, s=1) 
plt.xlabel("True values")
plt.ylabel("Predictions")
plt.show()

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(dep_feat_test, dep_feat_predMLR)))
print('R2:', metrics.r2_score(dep_feat_test, dep_feat_predMLR))
print('Mean Absolute Error: ', metrics.mean_absolute_error(dep_feat_test, dep_feat_predMLR))

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn import metrics


SVR_predictions = cross_val_predict(SVR(),indep_feat,dep_feat, cv=2 )
SVR_MAE_accuracy = metrics.mean_absolute_error(dep_feat, SVR_predictions)
print("Cross-Predicted MAE Accuracy for SVR:",SVR_MAE_accuracy)

SVR_RMSE_accuracy = np.sqrt(metrics.mean_squared_error(dep_feat, SVR_predictions))
print("Cross-Predicted RMSE Accuracy for SVR:",SVR_RMSE_accuracy)

RFR_predictions = cross_val_predict(RandomForestRegressor(),indep_feat,dep_feat, cv=20 )
RFR_MAE_accuracy = metrics.mean_absolute_error(dep_feat, RFR_predictions)
print("Cross-Predicted MAE Accuracy for RFR:",RFR_MAE_accuracy)

RFR_RMSE_accuracy = np.sqrt(metrics.mean_squared_error(dep_feat, RFR_predictions))
print("Cross-Predicted RMSE Accuracy for RFR:",RFR_RMSE_accuracy)

MLR_predictions = cross_val_predict(LinearRegression(),indep_feat,dep_feat, cv=2 )
MLR_MAE_accuracy = metrics.mean_absolute_error(dep_feat, MLR_predictions)
print("Cross-Predicted MAE Accuracy for MLR:",MLR_MAE_accuracy)

MLR_RMSE_accuracy = np.sqrt(metrics.mean_squared_error(dep_feat, MLR_predictions))
print("Cross-Predicted RMSE Accuracy for MLR:",MLR_RMSE_accuracy)



