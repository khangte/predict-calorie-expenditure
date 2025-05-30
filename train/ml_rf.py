import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import time
import json
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from preprocess.preprocess import load_and_preprocess
from utils.evaluations import evaluate

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 📦 데이터 로딩 및 전처리
print("📥 데이터 불러오는 중...")
X, y, X_test, test_ids = load_and_preprocess()
print("✅ 데이터 전처리 완료")

# 🔀 학습/검증 데이터 분할
print("📊 학습/검증 데이터 분할 중...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ 학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_val.shape}")

# 🔧 Best 파라미터 불러오기
print("⚙️ 베스트 파라미터 로딩 중...")
with open("data/best_params_randomforest.json", "r") as f:
    best_params = json.load(f)
print("✅ 불러온 파라미터:", best_params)

# ⏱ 모델 학습 시작
start_time = time.time()
print("🚀 Random Forest 모델 학습 시작...")
rf = RandomForestRegressor(**best_params)
rf.fit(X_train, y_train)

model_name, _, model = evaluate("Random Forest", rf, X_val, y_val)
end_time = time.time()
print(f"✅ 모델 학습 완료 (소요 시간: {end_time - start_time:.2f}초)")

# 🔮 테스트 데이터 예측
print("🔮 테스트 데이터 예측 중...")
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)
print("✅ 예측 완료")

# 💾 제출 파일 저장
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': test_pred
})

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"submissions/submission_{model_name.replace(' ', '_')}_{current_time}.csv"
submission.to_csv(filename, index=False)
print(f"📁 제출 완료: {filename}")
