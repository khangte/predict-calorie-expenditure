# optuna_tune_randomforest.py - Random Forest 하이퍼파라미터 튜닝
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from preprocess.preprocess import load_and_preprocess
import time

# 데이터 로딩 및 전처리
X, y, _, _ = load_and_preprocess()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 평가 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds_log = model.predict(X_valid)
    preds = pd.Series(preds_log).apply(lambda x: max(x, 0))  # log 값 보정
    score = mean_squared_log_error(y_valid, preds) ** 0.5
    return score

# Optuna 튜닝
start_time = time.time()
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
elapsed_time = time.time() - start_time

# 결과 출력 및 저장
print("✅ Best RMSLE:", study.best_value)
print("✅ Best Parameters:", study.best_params)
print(f"⏱ 튜닝 소요 시간: {elapsed_time:.2f}초")

# 저장
with open("data/best_params_randomforest.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
