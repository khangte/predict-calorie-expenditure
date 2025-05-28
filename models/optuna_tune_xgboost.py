import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from preprocess.preprocess import load_and_preprocess

# 전처리
X, y, X_test, test_ids = load_and_preprocess()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_val_actual = np.expm1(y_val)

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

# 튜닝 시작 시간
start_time = time.time()

# XGBoost 튜닝
print("\n🎯 XGBoost 튜닝 시작...")
study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=30)

# 튜닝 종료 시간 및 소요 시간 출력
end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏱️ 튜닝 소요 시간: {elapsed_time:.2f}초")

# 결과 출력 및 저장
print("Best RMSLE:", study.best_value)
print("Best Parameters:", study.best_params)

with open("data/best_params_xgb.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
print("✅ XGBoost 최적 파라미터 저장 완료")
