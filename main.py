from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from useful_package import *
from sklearn.model_selection import train_test_split


X = 10 + np.random.randn(1000, 1)
y = [[hyperbola(x) for x in X],
     [polynom_3(x) for x in X],
     ]
splits = []

for _y in y:
    X_train, X_test, y_train, y_test = train_test_split(
        X, _y, test_size=0.33, random_state=42)
    splits.append((X_train, X_test, y_train, y_test))
    
model = RandomForestRegressor()

for X_train, X_test, y_train, y_test in splits:
    model.fit(X_train, y_train)
    preds = model.predict(y_test)
    print("MSE: ", mean_squared_error(y_test, preds))
