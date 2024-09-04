import pandas as pd
data = pd.read_csv(r"C:\Users\sansk\Downloads\water_potability.csv")
print(data.head())
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 10))
p = sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

data.info()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
data_imputed = imputer.fit_transform(data)

# Convert to DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Split the data into features and target
X = data_imputed.drop('Potability', axis=1)
y = data_imputed['Potability']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

from sklearn import metrics
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)

svc_pred = svc_model.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, svc_pred)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test, svc_pred))

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
print("Test Accuracy =", format(metrics.accuracy_score(y_test, predictions)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

