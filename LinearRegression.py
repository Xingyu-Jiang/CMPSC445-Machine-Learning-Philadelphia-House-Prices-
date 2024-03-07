import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('random_sample_data.csv')

# Handle missing values if any
data.fillna(0, inplace=True)

# Split the dataset into features and target variable
X = data.drop(columns=['market_value'])
y = data['market_value']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a feature engineering pipeline
feature_engineering_pipeline = Pipeline([
    ('polynomial_features', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95))  # Retain 95% of variance
])

# Apply feature engineering to training and testing sets
X_train_fe = feature_engineering_pipeline.fit_transform(X_train)
X_test_fe = feature_engineering_pipeline.transform(X_test)

# Define LinearRegression model
model = LinearRegression()

# No hyperparameters to tune for Linear Regression

# Fit the model
model.fit(X_train_fe, y_train)

# Make predictions
predictions = model.predict(X_test_fe)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)
