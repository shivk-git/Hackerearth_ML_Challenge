import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# change object data into int or float
from sklearn.preprocessing import LabelEncoder
def Change_obj_type(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data



## training data
train_data = pd.read_csv("Dataset/Train.csv")
#print(train_data.head(10))
# drop the some columns
train_data = train_data.drop(['Employee_ID','Age','Relationship_Status'],axis=1)
#print(train_data.info())

sns.pairplot(train_data)
# change object to int or float
train_data = Change_obj_type(train_data)

# correlation plot
from scipy.stats import norm 
corr_m = train_data.corr()
f, ax = plt.subplots(figsize =(10, 9)) 
sns.heatmap(corr_m, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
print(corr_m['Attrition_rate'])


# trainig values of x and y
X_train = train_data.iloc[:,1:-1].values
Y_train = train_data.iloc[:,-1].values.reshape(-1,1)


# replace nan values with mean values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

impute_X = imputer.fit(X_train)
X_train = impute_X.transform(X_train)
#print(X_train.shape)

impute_Y = imputer.fit(Y_train)
Y_train = impute_Y.transform(Y_train)


## test data
tst_data = pd.read_csv("Dataset/Test.csv")
#print(test_data.info())
test_data = tst_data.drop(['Employee_ID','Age','Relationship_Status'],axis=1)

# change object to int or float
test_data = Change_obj_type(test_data)
        
        
X_test = test_data.iloc[:,:-1].values

#replace nan with mean values
imput_X = imputer.fit(X_test)
X_test = imput_X.transform(X_test)
#print(X_test.shape)

# change data into MinMaxScaler format
from sklearn.preprocessing import MinMaxScaler
# trainig set
stdx = MinMaxScaler()
stdx.fit(X)
X = stdx.transform(X)

stdy = StandardScaler()
stdy.fit(Y)
Y = stdy.transform(Y)
#X, Y

# test set
X_test = stdx.transform(X_test)
X_test


## 1.Prediction using artificial neural network (ANN)
import keras
import keras.backend as kb
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(32, input_dim=9, kernel_initializer="uniform", activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 32, kernel_initializer="uniform", activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 1, kernel_initializer="uniform"))
model.summary()

model.compile(loss='mse', optimizer = 'adam', metrics=['mse','mae'])

epochs_hist = model.fit(X, Y, epochs=20, batch_size=20, validation_split=0.8)

# values prediction
y_keras = model.predict(X_test)
y_keras = stdy.inverse_transform(y_keras)
y_keras


## 2.Using XGRoot
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',colsample_bytree = 1, learning_rate = 0.0335, max_depth = 5, alpha = 1, n_estimators=100)
xg_reg.fit(X,Y)
from xgboost import plot_importance
import matplotlib.pyplot as plt

# plot feature importance
plot_importance(xg_reg)
plt.show()
#xg_reg.feature_importances_
# values prediction
y_xg = xg_reg.predict(X_test)
y_xg = stdy.inverse_transform(y_xg)
y_xg



## 3.using state model
import statsmodels.api as sm
sm_model = sm.OLS(Y,X, rho=18).fit()
sm_model.params
# values prediction
y_sm = sm_model.predict(X_test)
y_sm = stdy.inverse_transform(y_sm)
y_sm


## 4.Using rendom forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=20)
rf.fit(X,Y.ravel())
# values prediction
y_rf = rf.predict(X_test)
y_rf = stdy.inverse_transform(y_rf)
y_rf


## 5.Using RidgeCV
from sklearn.linear_model import RidgeCV
Rid = RidgeCV(alphas = np.arange(0.1,100,1), fit_intercept=True)
Rid.fit(X,Y)
# values prediction
y_rid = Rid.predict(X_test)
y_rid = stdy.inverse_transform(y_rid)
y_rid


## 5.Using Lesso
from sklearn.linear_model import Lesso
les = Lasso(alpha=0.3, normalize=True)
les.fit(X,Y)
# values prediction
y_les = Rid.predict(X_test)
y_les = stdy.inverse_transform(y_les)
y_les


## 6. Using BayesianRidge, ARDRegression, LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LinearRegression
#compute_score=True
lr = BayesianRidge(compute_score=True)
lr.fit(X,Y.ravel())
# values prediction
y_lr = lr.predict(X_test)
y_lr = stdy.inverse_transform(y_lr)
y_lr


## 7. Using Spport vector machine
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', gamma='scale', tol=0.0000000000001, epsilon=0.2)
regressor.fit(X, Y.ravel())
# values prediction
y_svm = regressor.predict(X_test)
y_svm = stdy.inverse_transform(y_svm)
y_svm


# save file
data = pd.DataFrame()
data['Employee_ID'] = tst_data['Employee_ID']
data['Attrition_rate'] = y # y = y_svm, y_svm, y_rid, y_rf, y_sm, y_xg, y_keras, y_les
y_pred

data = data.to_csv('Emp_submission.csv',mode='w',index=False)
