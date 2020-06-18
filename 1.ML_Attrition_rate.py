import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
train_data = train_data.drop(['Employee_ID','Age','Relationship_Status','VAR4','VAR7'],axis=1)
#print(train_data.info())

train_data = Change_obj_type(train_data)

X_train = train_data.iloc[:,1:-1].values
Y_train = train_data.iloc[:,-1].values.reshape(-1,1)
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
test_data = tst_data.drop(['Employee_ID','Age','Relationship_Status','VAR4','VAR7'],axis=1)

test_data = Change_obj_type(test_data)
        
X_test = test_data.iloc[:,:-1].values
#replace nan with mean values
imput_X = imputer.fit(X_test)
X_test = imput_X.transform(X_test)
#print(X_test.shape)


import statsmodels.api as sm
model = sm.OLS(Y_train,X_train).fit()
#model.summary()

y_pred = model.predict(X_test)

data = pd.DataFrame()
data['Employee_ID'] = tst_data['Employee_ID']
data['Attrition_rate'] = y_pred
y_pred

data.to_csv('Emp_submission.csv')
