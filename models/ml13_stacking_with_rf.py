# ml8_stacking.py - Stacking Ensemble 기반 예측
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from preprocess.preprocess import load_and_preprocess
from utils.evaluations import evaluate

# 전처리
X, y, X_test, test_ids = load_and_preprocess()

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

with open("data/best_params_randomforest.json") as f:
    best_params_rf = json.load(f)
best_params_rf["n_jobs"] = -1

# k-fold 개수 지정
fold_cnt = 5

# KFold 설정
kf = KFold(n_splits=fold_cnt, shuffle=True, random_state=42)

# OOF & Test 예측 저장
oof_cat, oof_lgb, oof_xgb, oof_rf = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_pred_cat, test_pred_lgb, test_pred_xgb, test_pred_rf = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n🔄 Fold {fold+1}/{fold_cnt}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val = X.iloc[val_idx]

    # CatBoost
    cat = CatBoostRegressor(**best_params_cat)
    cat.fit(X_train, y_train, verbose=0)
    oof_cat[val_idx] = cat.predict(X_val)
    test_pred_cat.append(cat.predict(X_test))

    # LightGBM
    lgb = LGBMRegressor(**best_params_lgb)
    lgb.fit(X_train, y_train)
    oof_lgb[val_idx] = lgb.predict(X_val)
    test_pred_lgb.append(lgb.predict(X_test))

    # XGBoost
    xgb = XGBRegressor(**best_params_xgb)
    xgb.fit(X_train, y_train)
    oof_xgb[val_idx] = xgb.predict(X_val)
    test_pred_xgb.append(xgb.predict(X_test))

    # RandomForest
    rf = RandomForestRegressor(**best_params_rf)
    rf.fit(X_train, y_train)
    oof_rf[val_idx] = rf.predict(X_val)
    test_pred_rf.append(rf.predict(X_test))

# 메타모델 학습
stack_X = np.vstack([oof_cat, oof_lgb, oof_xgb, oof_rf]).T
stack_X_test = np.vstack([
    np.mean(test_pred_cat, axis=0),
    np.mean(test_pred_lgb, axis=0),
    np.mean(test_pred_xgb, axis=0),
    np.mean(test_pred_rf, axis=0)
]).T

meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta_model.fit(stack_X, y)

# 평가 지표 출력
evaluate("Stacking Ensemble (Ridge)", model=meta_model, X_val=stack_X, y_val_log=y)

stacked_pred_log = meta_model.predict(stack_X_test)
stacked_pred = np.expm1(stacked_pred_log)

# 제출 파일 생성
submission = pd.DataFrame({
    "id": test_ids,
    "Calories": stacked_pred
})

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_stacking_kf{fold_cnt}_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"\n✅ 제출 파일 저장 완료: {filename}")
