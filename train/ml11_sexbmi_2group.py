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

# 성별 + BMI 2구간 그룹화 (총 4개 그룹)
def get_bmi_group(sex, bmi):
    if bmi < 25:
        return f"{sex}_low"
    else:
        return f"{sex}_high"

# 데이터 로딩 및 그룹 생성
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train["BMI"] = train["Weight"] / (train["Height"] / 100) ** 2
test["BMI"] = test["Weight"] / (test["Height"] / 100) ** 2
train["group"] = train.apply(lambda row: get_bmi_group(row["Sex"], row["BMI"]), axis=1)
test["group"] = test.apply(lambda row: get_bmi_group(row["Sex"], row["BMI"]), axis=1)

# 공통 설정
numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
w_cat, w_lgb, w_xgb = 0.3364, 0.3326, 0.3310

# 최적 파라미터 로딩
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

# 그룹별 처리
all_preds = []
rmsle_list = []

for group in sorted(train['group'].unique()):
    print(f"\n📂 그룹: {group}")
    train_group = train[train["group"] == group].copy()
    test_group = test[test["group"] == group].copy()

    # 스케일링
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(train_group[numeric_features]), columns=numeric_features)
    y = np.log1p(train_group['Calories'])
    X_test = pd.DataFrame(scaler.transform(test_group[numeric_features]), columns=numeric_features)

    # 검증용 RMSLE 계산
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    cat = CatBoostRegressor(**best_params_cat).fit(X_train, y_train)
    lgb = LGBMRegressor(**best_params_lgb).fit(X_train, y_train)
    xgb = XGBRegressor(**best_params_xgb).fit(X_train, y_train)

    y_pred_log = w_cat * cat.predict(X_val) + w_lgb * lgb.predict(X_val) + w_xgb * xgb.predict(X_val)
    y_val_actual = np.expm1(y_val)
    y_pred_actual = np.expm1(y_pred_log)
    rmsle = np.sqrt(mean_squared_log_error(y_val_actual, y_pred_actual))
    print(f"📊 RMSLE: {rmsle:.5f}")
    rmsle_list.append(rmsle)

    # 전체 학습 후 테스트 예측
    cat = CatBoostRegressor(**best_params_cat).fit(X, y)
    lgb = LGBMRegressor(**best_params_lgb).fit(X, y)
    xgb = XGBRegressor(**best_params_xgb).fit(X, y)

    pred_log = w_cat * cat.predict(X_test) + w_lgb * lgb.predict(X_test) + w_xgb * xgb.predict(X_test)
    pred_final = np.expm1(pred_log)

    df = pd.DataFrame({
        "id": test_group["id"].values,
        "Calories": pred_final
    })
    all_preds.append(df)

# 병합 및 저장
submission = pd.concat(all_preds).sort_values("id")
filename = f"submissions/submission_sexbmi_2group_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
submission.to_csv(filename, index=False)

# 전체 평균 RMSLE 출력
mean_rmsle = np.mean(rmsle_list)
print(f"\n📈 전체 평균 RMSLE: {mean_rmsle:.5f}")
print(f"✅ 제출 파일 저장 완료: {filename}")
