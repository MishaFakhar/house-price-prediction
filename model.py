import numpy as np
import pandas as pd

# Load the dataset (replace with your dataset path)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(url, delim_whitespace=True, names=columns)

# Features and target
X = data.drop('MEDV', axis=1).values
y = data['MEDV'].values

# Adding a bias term to features
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance

# Train the model using Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

def predict_price(features):
    features = np.array(features).reshape(1, -1)
    features_b = np.c_[np.ones((features.shape[0], 1)), features]  # add x0 = 1
    return features_b.dot(theta_best)[0]
