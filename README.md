# 📊 Predict Calorie Expenditure  
[Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5/)

---

## 최종 결과
- Rank  : **365**/4316
- Score : **0.05865**
- 선정 실험 : 8번째 실험

### 실험 요약
- **베이스 모델**: CatBoost, LightGBM, XGBoost  
- **메타 모델**: RidgeCV  
- **Stacking 방식**: K-Fold 기반 OOF stacking (5-Fold)  
- **하이퍼파라미터 최적화**: Optuna를 이용해 각 베이스 모델의 최적 파라미터 사전 탐색  
- **파라미터 적용 방식**: JSON 파일로 저장된 Optuna 결과 불러와 각 모델에 적용  
- **평가 지표**: 로그 스케일 RMSLE 사용  
- **예측 후처리**: `np.expm1()`으로 로그 예측값을 원래 스케일로 복원  
- **성능 결과**: 거의 가장 낮은 RMSLE을 기록한 최고의 성능 모델  

---

### 🔬 첫 번째 실험 (`ml.py`)
- 7가지 모델 비교: 다중회귀, 결정트리, 경사하강법, LightGBM, XGBoost, CatBoost, 딥러닝
- 최종 선택 모델: **CatBoost**
- **RMSLE**: `0.0595`  
- **Score**: `0.05755`  
- 📄 제출파일: `submission_CatBoost_20250526_163800.csv`

---

### 🔬 두 번째 실험 (`ml2.py` + `optuna_tune_catboost.py`)
- Optuna를 이용해 CatBoost 하이퍼파라미터 튜닝
- 30회 모델 학습 → 최적 조합 선택
- **RMSLE**: `0.0592`  
- **Score**: `0.05739`  
- 📄 제출파일: `submission_catboost_optuna_20250526_173823.csv`

---

### 🔬 세 번째 실험 (`ml3.py`)
- `BMI` 파생 변수 추가
- CatBoost vs LightGBM 비교  
- 최종 선택: **CatBoost**
- **RMSLE**:  
  - CatBoost: `0.05919`  
  - LightGBM: `0.06211`  
- **Score**: `0.05746`  
- 📄 제출파일: `submission_bmi_catboost_20250526_222929.csv`

---

### 🔬 네 번째 실험 (`ml4.py`)
- BMI 변수 유지 + Optuna 최적 하이퍼파라미터 적용
- CatBoost, LightGBM, XGBoost 비교  
- 최종 선택: **CatBoost**
- **RMSLE**:  
  - CatBoost: `0.05919`  
  - LightGBM: `0.05995`  
  - XGBoost: `0.06022`  
- **Score**: `0.05746`  
- 📄 제출파일: `submission_bmi_catboost_20250527_120910.csv` (미제출)

---

### 🔬 다섯 번째 실험 (`ml5.py`)
- 3개 모델 예측 평균 블렌딩
- **Score**: `0.05713`  
- 📄 제출파일: `submission_catboost_blended_20250526_1759.csv`

---

### 🔬 여섯 번째 실험 (`ml6.py`)
- 모델별 RMSLE 기반 가중 평균 블렌딩  
- CatBoost: `0.05919`, LGBM: `0.05995`, XGBoost: `0.06022`
- **Score**: `0.05713`  
- 📄 제출파일: `submission_weighted_blend_20250527_122812.csv`

---

### 🔬 일곱 번째 실험 (`ml7_kfold_blend.py`)
- KFold 기반 모델 예측 + RMSLE 정규화 가중 평균
- **Score**: `0.05703` ✅ 최고 성능  
- 📄 제출파일: `submission_kfold_blend_20250527_124155.csv`

---

### 🔬 여덟 번째 실험 (`ml8_stacking.py`)
- KFold OOF → RidgeCV 메타모델로 Stacking
- **Score**: `0.05698` 🏆 최고 성능  
- 📄 제출파일: `submission_stacking_20250527_124906.csv`

---

### 🔬 아홉 번째 실험 (`ml9_gender_split.py`)
- 성별 기반 분할 학습 (남성 RMSLE 0.06851로 성능 하락)
- 📄 미제출

---

### 🔬 열 번째 실험 (`ml10_sexbmi_split.py`)
- 성별 + BMI 구간 조합 (8개 그룹)
- CatBoost, LGBM, XGBoost + 가중 평균
- **Score**: `0.05754`  
- 📄 제출파일: `submission_sexbmi_split_20250527_142906.csv`

---

### 🔬 열한 번째 실험 (`ml11_sexbmi_2group.py`)
- 성별 + BMI 2구간 → 4개 그룹 분할
- **RMSLE**: `0.05948`  
- 📄 미제출

---

### 🔬 열두 번째 실험 (`ml8_stacking.py`)
- KFold 5 → 10으로 확장
- RidgeCV 기반 스태킹  
- **RMSLE**: `0.0592`  
- 📄 제출파일: `submission_stacking_20250528_XXXXXX.csv` (미제출)

---

### 🔬 열세 번째 실험 (`ml13_stacking_with_rf.py`)
- 기존 모델에 `RandomForest` 추가
- 메타모델: `RidgeCV`
- **RMSLE**: `0.05917`  
- 📄 제출파일: `submission_stacking_rf_YYYYMMDD_HHMMSS.csv`

---

### 🔬 열네 번째 실험 (`ml_pipeline_stacking.py`)
- 메타모델로 `LGBMRegressor` 사용
- 조합 특성 추가 (`HR_Duration`, `BT_Duration`)
- **RMSLE**: `0.0597`  
- 📄 제출파일: `submission_stacking_kf5_YYYYMMDD_HHMMSS.csv`

---

### 🔬 열다섯 번째 실험 (`ml_pipeline_stacking_improved.py`)
- 예측값 차이 피처 (`lgb-cat`, `xgb-cat`) 추가
- 메타모델: `RidgeCV`
- **RMSLE**: `0.0593`  
- **Score**: `0.05708`  
- 📄 제출파일: `submission_stacking_improved_20250529_162009.csv`

---

### 🔬 열여섯 번째 실험 (`ml_pipeline_stacking_improved.py`)
- 조합 피처: `BMI`, `Temp_per_Duration` 추가
- 메타모델: `RidgeCV`
- **RMSLE**: `0.0595`  
- 📄 제출파일: (미제출)

---

### 🔬 열여덟 번째 실험

- **파일명**: `ml_pipeline_stacking.py`
- **주요 특징**:
  - `sklearn.ensemble.StackingRegressor`를 이용한 간단한 Stacking 구현
  - 별도 KFold 없이 내부적으로 전체 데이터를 기반으로 메타모델 학습
  - 메타모델로 `RidgeCV` 사용
  - 베이스 모델: `CatBoost`, `LightGBM`, `XGBoost`
  - 메타 피처로 각 모델의 예측값만 사용

- **변경 사항**:
  - 기존 수동 KFold 기반 stacking → `StackingRegressor`로 변경
  - `KFold` 분할 없이 전체 데이터를 학습해 stacking 구조 간소화
  - `RidgeCV`는 기본 설정으로 사용 (알파 자동 탐색)

- **성능 결과**:
  - **RMSLE** (훈련 OOF 기준): **0.0556** ✅ (기존보다 낮은 수치)
  - **Score** (제출 결과): **0.05711** ⚠️
  - 내부 점수는 낮았으나, 제출 점수는 더 높음 → 과적합 가능성

- **제출 파일**:
  - `submission_stacking_20250530_XXXXXX.csv`

- **해석**:
  - 훈련 성능(RMSLE)은 매우 낮았지만, 실제 테스트 데이터에서는 오히려 성능 저하
  - `StackingRegressor`가 전체 훈련 데이터를 사용하며 메타모델이 과적합된 것으로 판단
  - 향후 개선 필요 사항:
    - `cv=5` 옵션을 추가하여 내부적으로 OOF 예측 기반 학습 유도
    - `passthrough=True`를 사용하여 원본 피처도 함께 전달
    - 또는 기존 수동 KFold 기반 stacking 방식으로 회귀
