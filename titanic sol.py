import pandas as pd
import matplotlib as mp
import numpy as np
data=pd.read_csv('train.csv')
data2=pd.read_csv('test.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
X2=data2.iloc[:,:].values
from sklearn.preprocessing import Imputer, OneHotEncoder
imputer = Imputer(missing_values ='NaN'  , strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, :])
X[:, :]=imputer.transform(X[:, :])
onehotencoder=OneHotEncoder(categorical_features=[1,6])
X=onehotencoder.fit_transform(X).toarray()
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.linear_model import LogisticRegression
c = LogisticRegression(random_state =0)
c.fit(X,Y) 
X2[:, :]=imputer.transform(X2[:, :])
X2=onehotencoder.fit_transform(X2).toarray()
X2 = sc_X.fit_transform(X2)
y_pred=c.predict(X2)
df=pd.DataFrame(y_pred)
df.to_csv('sol.csv')