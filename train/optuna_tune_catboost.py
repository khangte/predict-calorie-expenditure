# optuna_tune.py - 최적 파라미터 탐색 후 저장
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import json
import time
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

from preprocess.preprocessing_v2 import load_and_preprocess

# 데이터 로딩
X, y, _, _ = load_and_preprocess()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1000),
        "depth": trial.suggest_int("depth", 4, 8),  # 깊은 트리 제한
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),  # 너무 낮거나 높지 않게
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),  # 규제 강도 높여서 일반화 유도
        "random_seed": 42,
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
        "verbose": False
    }

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True  # early stopping 이후 최고 성능 시점의 모델 저장
    )

    y_pred = np.expm1(model.predict(X_val))
    y_true = np.expm1(y_val)

    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# 튜닝 시작 시간
start_time = time.time()

# 스터디 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# 튜닝 종료 시간 및 소요시간 출력
end_time = time.time()
elapsed_time = end_time - start_time
print(f"튜닝 소요 시간: {elapsed_time:.2f}초")

# 결과 출력 및 저장
print("Best RMSLE:", study.best_value)
print("Best Parameters:", study.best_params)

with open("data/best_params_catboost.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

print("\u2705 저장 완료:", study.best_params)
