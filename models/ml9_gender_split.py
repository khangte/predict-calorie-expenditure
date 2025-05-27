# ml9_gender_split.py - 성별 기반 분리 학습 및 예측 결합 + RMSLE 계산

import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from datetime import datetime

# 데이터 로딩
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 성별 분할
groups = ['male', 'female']
all_preds = []
rmsle_scores = []

for gender in groups:
    print(f"\n📂 성별: {gender.upper()}")

    # 필터링
    train_gender = train[train['Sex'] == gender].copy()
    test_gender = test[test['Sex'] == gender].copy()

    # 수치형 특성 및 스케일링
    numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    scaler = StandardScaler()
    X_full = pd.DataFrame(scaler.fit_transform(train_gender[numeric_features]), columns=numeric_features)
    y_full = np.log1p(train_gender['Calories'])
    X_test = pd.DataFrame(scaler.transform(test_gender[numeric_features]), columns=numeric_features)

    # 훈련/검증 분리 (RMSLE 평가용)
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    # 파라미터 로딩
    with open("data/best_params_catboost.json") as f:
        best_params_cat = json.load(f)
    best_params_cat["random_seed"] = 42
    best_params_cat["logging_level"] = "Silent"

    with open("data/best_params_lgb.json") as f:
        best_params_lgb = json.load(f)
    best_params_lgb["random_state"] = 42

    with open("data/best_params_xgb.json") as f:
        best_params_xgb = json.load(f)
    best_params_xgb["random_state"] = 42

    # 모델 학습 (fold별 검증 포함)
    cat = CatBoostRegressor(**best_params_cat).fit(X_train, y_train, verbose=0)
    lgb = LGBMRegressor(**best_params_lgb).fit(X_train, y_train)
    xgb = XGBRegressor(**best_params_xgb).fit(X_train, y_train)

    # RMSLE 평가
    pred_val_log = (0.3364 * cat.predict(X_val) + 0.3326 * lgb.predict(X_val) + 0.3310 * xgb.predict(X_val))
    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(pred_val_log)))
    rmsle_scores.append((gender, rmsle))
    print(f"🔍 {gender.upper()} RMSLE: {rmsle:.5f}")

    # 전체 학습 후 예측
    cat.fit(X_full, y_full, verbose=0)
    lgb.fit(X_full, y_full)
    xgb.fit(X_full, y_full)
    pred_log = (0.3364 * cat.predict(X_test) + 0.3326 * lgb.predict(X_test) + 0.3310 * xgb.predict(X_test))
    pred_final = np.expm1(pred_log)

    df = pd.DataFrame({
        "id": test_gender["id"].values,
        "Calories": pred_final
    })
    all_preds.append(df)

# 전체 병합 후 저장
submission = pd.concat(all_preds).sort_values("id")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_gender_split_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\n✅ 성별 분리 제출 파일 저장 완료: {filename}")

# 전체 RMSLE 요약 출력
print("\n📊 성별별 RMSLE:")
for gender, score in rmsle_scores:
    print(f"{gender}: {score:.5f}")
