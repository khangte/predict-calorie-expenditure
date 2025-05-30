# pipeline/ml_pipeline.py
# 필요한 라이브러리 임포트
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
import joblib

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# 데이터 불러오기
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# log1p 적용 (RMSLE 평가 기준)
train["Calories"] = np.log1p(train["Calories"])

# 컬럼 선택
drop_cols = ['id', 'Calories']
numeric_feats = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_feats = train.select_dtypes(include=['object']).columns.tolist()

numeric_feats = [col for col in numeric_feats if col not in drop_cols]
categorical_feats = [col for col in categorical_feats if col not in drop_cols]
main_features = numeric_feats + categorical_feats

# 훈련/검증 데이터 분할
y = train['Calories']
X = train[main_features]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 수치형 변수 전처리 파이프라인 구성
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

# 범주형 변수 전처리 파이프라인 구성
categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 전처리 파이프라인 통합
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, categorical_feats)
])

# 모델 후보군 정의
models = {
    # 'Ridge': Ridge(),
    # 'Lasso': Lasso(),
    # 'ElasticNet': ElasticNet(),
    # 'XGBoost': xgb.XGBRegressor(tree_method='hist', random_state=42),
    # 'LightGBM': lgb.LGBMRegressor(random_state=42, verbosity=-1),
    # 'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
    # 'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    # 'HistGB': HistGradientBoostingRegressor(random_state=42),
    # 'KNN': KNeighborsRegressor(n_neighbors=5),
    # 'SVR' : SVR(C=1.0, epsilon=0.1),
    'MLP' : MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'ExtraTrees' : ExtraTreesRegressor(n_estimators=200, random_state=42)

}

# 모델 학습 및 평가 (RMSLE)
results = {}
best_model = None
best_rmsle = float('inf')

for name, model in models.items():
    print(f'\n==== {name} ====')
    try:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('reg', model)
        ])

        pipe.fit(X_train, y_train)

        y_val_pred_log = pipe.predict(X_val)
        y_val_pred = np.expm1(y_val_pred_log)
        y_val_actual = np.expm1(y_val)

        rmsle = np.sqrt(mean_squared_log_error(y_val_actual, y_val_pred))
        print(f'Validation RMSLE: {rmsle:.4f}')

        results[name] = rmsle

        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_model = pipe

    except Exception as e:
        print(f"Error: {str(e)}")

# 최종 결과 비교
print("\n==== Final Results ====")
results_df = pd.DataFrame(results.items(), columns=['Model', 'RMSLE'])
results_df = results_df.sort_values('RMSLE')
print(results_df)

# 테스트 예측
print("\n==== Test Data Prediction ====")
X_test = test[main_features]
test_pred_log = best_model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# 제출 파일 생성
submission = pd.DataFrame({
    'id': test['id'],
    'Calories': test_pred
})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_filename = f'submissions/submission_{timestamp}.csv'
submission.to_csv(submission_filename, index=False)
print(f"\nSubmission 파일이 생성되었습니다: {submission_filename}")

# 최고 성능 모델 저장
model_filename = f'models/best_model_{timestamp}.joblib'
joblib.dump(best_model, model_filename)
print(f"최고 성능 모델이 저장되었습니다: {model_filename}")
