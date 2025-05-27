# optuna_tune_lgb_xgb.py - LightGBM & XGBoost 하이퍼파라미터 튜닝 및 저장

import optuna
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# 데이터 로딩 및 전처리
train = pd.read_csv("train.csv")
train["BMI"] = train["Weight"] / ((train["Height"] / 100) ** 2)
train = pd.concat([train.drop('Sex', axis=1), pd.get_dummies(train['Sex'], prefix='Sex')], axis=1)

numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[numeric_features])
X = pd.concat([
    pd.DataFrame(X_scaled, columns=numeric_features),
    train[['Sex_female', 'Sex_male']]
], axis=1)
y = np.log1p(train['Calories'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_val_actual = np.expm1(y_val)

# LightGBM 튜닝 함수
def objective_lgb(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0),
        'random_state': 42
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred = np.expm1(model.predict(X_val))
    return np.sqrt(mean_squared_log_error(y_val_actual, pred))

# XGBoost 튜닝 함수
def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0),
        'random_state': 42
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    pred = np.expm1(model.predict(X_val))
    return np.sqrt(mean_squared_log_error(y_val_actual, pred))

# LightGBM 튜닝
print("\n🎯 LightGBM 튜닝 시작...")
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=30)
with open("best_params_lgb.json", "w") as f:
    json.dump(study_lgb.best_params, f)
print("✅ LightGBM 최적 파라미터 저장 완료")

# XGBoost 튜닝
print("\n🎯 XGBoost 튜닝 시작...")
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=30)
with open("best_params_xgb.json", "w") as f:
    json.dump(study_xgb.best_params, f)
print("✅ XGBoost 최적 파라미터 저장 완료")
