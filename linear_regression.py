import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
data = pd.read_csv('linear_regression/health_data_large.csv')

# Preprocess 'time' column
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek  # Monday=0, Sunday=6
data['month'] = data['time'].dt.month

# Define features (X) and target variables (y)
X = data[['hour', 'day_of_week', 'month']]
y = data[['Heartbeat', 'SpO2', 'Temperature']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the multiple regression model

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'multiple_regression_model.pkl')
