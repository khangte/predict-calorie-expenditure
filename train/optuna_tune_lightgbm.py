import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import json
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor, early_stopping
from preprocess.preprocess import load_and_preprocess

# 전처리
X, y, X_test, test_ids = load_and_preprocess()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_val_actual = np.expm1(y_val)

# LightGBM 튜닝 함수
def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42
    }

    model = LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(50)]
    )

    y_pred = model.predict(X_val)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_val, y_pred))

# 튜닝 시작 시간
start_time = time.time()

# LightGBM 튜닝
print("\n🎯 LightGBM 튜닝 시작...")
study = optuna.create_study(direction="minimize")
study.optimize(objective_lgb, n_trials=30)

# 튜닝 종료 시간 및 소요 시간 출력
end_time = time.time()
elapsed_time = end_time - start_time
print(f"⏱️ 튜닝 소요 시간: {elapsed_time:.2f}초")

# 결과 출력 및 저장
print("Best RMSLE:", study.best_value)
print("Best Parameters:", study.best_params)

with open("data/best_params_lgb.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
print("✅ LightGBM 최적 파라미터 저장 완료")
