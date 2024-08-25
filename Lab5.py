import pandas as pd
 import numpy as np
 from sklearn.model_selection import train_test_split
 from sklearn.linear_model import LinearRegression
 from sklearn.metrics import mean_squared_error, r2_score

datapath='/content/VSP - Java.xlsx'
df=pd.read_excel(datapath)
Xi=df[['Header_and_Main_declaration']]
X=df[['Header_and_Main_declaration','Incomprehensible_Code','Comprehensible_code_with_logical_errors','Comprehensible_code_with_syntax_errors']]
y=df[['Final_Marks']]

Xi_train, Xi_test, y_train, y_test = train_test_split(Xi,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(Xi_train,y_train)
y_test_pred=model.predict(Xi_test)
y_train_pred=model.predict(Xi_train)

def stats(y_test,y_pred):
  mse=mean_squared_error(y_test,y_pred)
  rmse=np.sqrt(mse)

  y_array_test=np.array(y_test)
  y_array_test_pred=np.array(y_test_pred)
  y_array_train=np.array(y_train)
  y_array_train_pred=np.array(y_test_pred)
  mape=np.mean(np.abs((y_test-y_pred)/y_test))*100

  r2=r2_score(y_test,y_pred)

  print('Mean Squared Error:',mse)
  print('R-squared:',r2)
  print('MAPE:',mape,'\n')

stats(y_test,y_test_pred)
stats(y_train,y_train_pred

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_test_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

stats(y_test,y_test_pred)
stats(y_train,y_train_pred)
