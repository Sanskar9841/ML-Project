import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv(r'C:\Users\sansk\Downloads\hr.csv')

# Drop rows with missing values or handle them if necessary
df = df.dropna()

# Separate features and target variable
X = df.drop(columns=['left'])
y = df['left']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

# Check for unexpected data types in numeric columns
print("Unexpected data types in numeric columns:\n", X[numeric_cols].dtypes)

# Define preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create pipelines with preprocessing and models
pipeline_logistic = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline_svm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the types of data in X_train after splitting
print("Data types in X_train before fitting:\n", X_train.dtypes)

# Fit and evaluate Logistic Regression model
pipeline_logistic.fit(X_train, y_train)
y_pred_logistic = pipeline_logistic.predict(X_test)

print("Logistic Regression Model Evaluation:")
print(confusion_matrix(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))

# Fit and evaluate SVM model
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)

print("SVM Model Evaluation:")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
