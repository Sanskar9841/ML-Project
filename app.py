
import pandas as pd
df = pd.read_csv(r"C:\Users\sansk\OneDrive\Desktop\ads.csv")
print(df)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png')
plt.show()

df.hist(bins=30, figsize=(10, 8))
plt.suptitle('Histogram of Features')
plt.savefig('histogram.png')
plt.show()

df.plot(kind='bar', x='TV', y='sales', figsize=(10, 8))
plt.title('Bargraph of TV vs Sales')
plt.savefig('bargraph.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(df['TV'], shade=True)
plt.title('KDE Plot of TV Ad Budget')
plt.savefig('kdeplot.png')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['TV']]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Linear Regression, Single Feature): {mse}')

X_multi = df[['TV', 'radio', 'newspaper']]  
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
lr_model_multi = LinearRegression()
lr_model_multi.fit(X_train, y_train)
y_pred_multi = lr_model_multi.predict(X_test)
mse_multi = mean_squared_error(y_test, y_pred_multi)
print(f'Mean Squared Error (Linear Regression, Multiple Features): {mse_multi}')

from sklearn.svm import SVR
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)

print(f'Mean Squared Error (SVR): {mse_svr}')
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=1)
X_new = selector.fit_transform(df.drop('sales', axis=1), df['sales'])
selected_feature = df.drop('sales', axis=1).columns[selector.get_support()][0]
print(f'Selected Feature: {selected_feature}')

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\sansk\OneDrive\Desktop\ads.csv")  
tv_budget = st.number_input('Enter TV ad budget')
X = df[['TV']]
y = df['sales']
model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[tv_budget]])

st.write(f'Predicted Sales: {prediction[0]}')
plt.figure(figsize=(10, 6))
sns.regplot(x='TV', y='sales', data=df)
plt.title('TV Ad Budget vs Sales')
st.pyplot(plt)
st.image('heatmap.png', caption='Heatmap')
st.image('histogram.png', caption='Histogram')
st.image('bargraph.png', caption='Bargraph')
st.image('kdeplot.png', caption='KDE Plot')
