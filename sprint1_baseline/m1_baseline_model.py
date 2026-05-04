import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- 1. Load Data ---
df = pd.read_csv('stream_pulse_customer_data.csv')

# --- 2. Separate Features and Target ---
X = df.drop(['Churn', 'CustomerID'], axis=1)
y = df['Churn']

# --- 3. Stratified Split (70/30) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# --- 4. Preprocessing Setup ---
numeric_features = ['Age', 'TenureMonths', 'MonthlyFees', 'TotalSpend', 'DeviceCount', 'SupportTickets']
categorical_features = ['PlanType', 'Region', 'PaymentMethod', 'StreamingService', 'AutoPay', 'FamilyPlan']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 5. The Pipeline ---
# This combines preprocessing and the model into one object
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# --- 6. Hyperparameter Tuning & 5-Fold Cross-Validation ---
# We use 'classifier__' because that is the name we gave the model in the Pipeline
param_grid = {
    'classifier__C': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# --- 7. Evaluation ---
y_pred = grid_search.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("\nModel Evaluation Report:\n")
print(classification_report(y_test, y_pred))

# --- 8. Save the Final Objects ---
# Requirement: Save the model and the pipeline. 
# Since our 'grid_search.best_estimator_' IS the pipeline, we save it as one.
joblib.dump(grid_search.best_estimator_, 'baseline_model.pkl')
# If your instructor specifically wants the preprocessor separate:
joblib.dump(preprocessor, 'preprocessing_pipeline.pkl')

print("Files saved: baseline_model.pkl and preprocessing_pipeline.pkl")
