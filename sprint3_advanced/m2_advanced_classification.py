# dataset link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(0)

df.drop('customerID', axis = 1, inplace = True)
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])
    
X = df.drop('Churn', axis = 1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds)
    results[name] = score
    print(f"{name} F1-Score: {score:.4f}")
    

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000), 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1',
    n_jobs=-1 
)

grid_search.fit(X_train, y_train)

optimized_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

y_pred_final = optimized_model.predict(X_test)
print("\nFinal Optimized Performance Report:")
print(classification_report(y_test, y_pred_final))

cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix: Optimized Logistic Regression')
plt.xlabel('Predicted Label (0=Stay, 1=Churn)')
plt.ylabel('Actual Label (0=Stay, 1=Churn)')
plt.show()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

model_filename = 'optimized_churn_model.pkl'
joblib.dump(optimized_model, model_filename)
