import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('bangladesh_student_performance.csv')
df.drop('date', axis=1, inplace=True)
df.drop('age', axis=1, inplace=True)

# Prepare features and target
X = df.drop('hsc_result', axis=1)
y = df['hsc_result']

# Convert object columns to strings to avoid StringDtype
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype(str)

# Identify feature types
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include='object').columns

# Create transformers
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=6,
        random_state=42
    ))
])

# Split data and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("gbr_model_fixed.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model recreated and saved as 'gbr_model_fixed.pkl'")
print("Test score:", model.score(X_test, y_test))
