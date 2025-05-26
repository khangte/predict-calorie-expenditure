# Predict Calorie Expenditure
[Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5/)

---

- 첫 번째 제출
    + ```ml.py```
        - [다중회귀, 결정트리, 경사하강법, LightGBM, XGBoost, CatBoost, 딥러닝]
        - 위 7가지 방법 중 RMSLE가 가장 낮은 모델 선택
    + model: CatBoost
    + RMSLE: 0.0595
    + score: **0.05755**
    + 제출파일: submission_CatBoost_20250526_163800.csv

- 두번째 제출
    + ```ml2.py```+```optuna_tune.py```
    + ```Optuna``` 사용
        - '첫번째 제출'에서 선정된 모델의 하이퍼파라미터를 자동으로 조합
        - 30번 모델 학습
        - 평가지표가 가장 낮은 최적의 하이퍼파라미터 조합을 찾음
    + RMSLE: 0.0592
    + score: **0.5739**
    + 제출파일: submission_catboost_optuna_20250526_173823.csv

