# ml7_kfold_blend.py - KFold 앙상블 기반 블렌딩 예측 제출

import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from preprocess import load_and_preprocess
from datetime import datetime

# 전처리
X, y, X_test, test_ids = load_and_preprocess()

# 모델 파라미터 로딩
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

# KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_preds = []
lgb_preds = []
xgb_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n🚀 Fold {fold+1}/5")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    # CatBoost
    cat_model = CatBoostRegressor(**best_params_cat)
    cat_model.fit(X_train, y_train, verbose=0)
    cat_preds.append(cat_model.predict(X_test))

    # LightGBM
    lgb_model = LGBMRegressor(**best_params_lgb)
    lgb_model.fit(X_train, y_train)
    lgb_preds.append(lgb_model.predict(X_test))

    # XGBoost
    xgb_model = XGBRegressor(**best_params_xgb)
    xgb_model.fit(X_train, y_train)
    xgb_preds.append(xgb_model.predict(X_test))

# 평균 앙상블
cat_pred_log = np.mean(cat_preds, axis=0)
lgb_pred_log = np.mean(lgb_preds, axis=0)
xgb_pred_log = np.mean(xgb_preds, axis=0)

# 정규화된 RMSLE 기반 가중치 적용
w_cat, w_lgb, w_xgb = 0.3364, 0.3326, 0.3310
blended_pred_log = (w_cat * cat_pred_log + w_lgb * lgb_pred_log + w_xgb * xgb_pred_log)
blended_pred = np.expm1(blended_pred_log)

# 제출 파일 저장
submission = pd.DataFrame({
    "id": test_ids,
    "Calories": blended_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_kfold_blend_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\n✅ 제출 파일 저장 완료: {filename}")
