# optuna_tune.py - 최적 파라미터 탐색 후 저장

import optuna
import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler

# 데이터 로딩
train = pd.read_csv("data/train.csv")
train = pd.concat([train.drop('Sex', axis=1), pd.get_dummies(train['Sex'], prefix='Sex')], axis=1)

numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[numeric_features])
X = pd.concat([
    pd.DataFrame(X_scaled, columns=numeric_features),
    train[['Sex_female', 'Sex_male']]
], axis=1)
y = np.log1p(train['Calories'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0),
        "random_seed": 42,
        "logging_level": "Silent"
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = np.expm1(model.predict(X_val))
    y_true = np.expm1(y_val)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

with open("data/best_params_catboost.json", "w") as f:
    json.dump(study.best_params, f)

print("\u2705 저장 완료:", study.best_params)
