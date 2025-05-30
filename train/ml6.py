# ml6.py - CatBoost, LightGBM, XGBoost 정규화된 RMSLE 가중 평균 블렌딩 예측 및 제출

import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from preprocess import load_and_preprocess
from datetime import datetime

# 공통 전처리 함수 호출
X, y, X_test, test_ids = load_and_preprocess()

# 최적 파라미터 불러오기
with open("data/best_params_catboost.json", "r") as f:
    best_params_cat = json.load(f)
best_params_cat["random_seed"] = 42
best_params_cat["logging_level"] = "Silent"

with open("data/best_params_lgb.json", "r") as f:
    best_params_lgb = json.load(f)
best_params_lgb["random_state"] = 42

with open("data/best_params_xgb.json", "r") as f:
    best_params_xgb = json.load(f)
best_params_xgb["random_state"] = 42

# 모델 학습
cat_model = CatBoostRegressor(**best_params_cat)
cat_model.fit(X, y)
cat_pred_log = cat_model.predict(X_test)

lgb_model = LGBMRegressor(**best_params_lgb)
lgb_model.fit(X, y)
lgb_pred_log = lgb_model.predict(X_test)

xgb_model = XGBRegressor(**best_params_xgb)
xgb_model.fit(X, y)
xgb_pred_log = xgb_model.predict(X_test)

# 정규화된 RMSLE 기준 가중 평균 블렌딩
w_cat, w_lgb, w_xgb = 0.3364, 0.3326, 0.3310
blended_pred_log = (w_cat * cat_pred_log + w_lgb * lgb_pred_log + w_xgb * xgb_pred_log)
blended_pred = np.expm1(blended_pred_log)

# 제출 파일 저장
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': blended_pred
})
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_weighted_blend_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\U0001F4C1 제출 파일 저장 완료: {filename}")
