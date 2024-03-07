import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('random_sample_data.csv')

# Handle missing values if any
data.fillna(0, inplace=True)

# Split the dataset into features and target variable
X = data.drop(columns=['market_value'])  # Features
y = data['market_value']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a feature engineering pipeline
feature_engineering_pipeline = Pipeline([
    ('polynomial_features', PolynomialFeatures(degree=2)),  # Create polynomial features
    ('scaler', StandardScaler()),  # Standardize features
    ('pca', PCA(n_components=0.95))  # Perform PCA for dimensionality reduction while retaining 95% variance
])

# Apply feature engineering to training and testing sets
X_train_fe = feature_engineering_pipeline.fit_transform(X_train)
X_test_fe = feature_engineering_pipeline.transform(X_test)

# Define GradientBoostingRegressor model
model = GradientBoostingRegressor(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_fe, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
predictions = best_model.predict(X_test_fe)

# Evaluate the best model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print('Best Model Hyperparameters:', grid_search.best_params_)
print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)
