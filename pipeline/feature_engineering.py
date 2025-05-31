# feature_engineering.py
import numpy as np
import pandas as pd


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    고급 피처 엔지니어링 적용 함수: 운동 생리학 기반 조합 피처 생성
    """

    # 결측치 방어 처리용 epsilon
    eps = 1e-6

    # 기본 조합 피처
    df['HR_Duration'] = df['Heart_Rate'] * df['Duration']
    df['BT_Duration'] = df['Body_Temp'] * df['Duration']
    df['Temp_per_Duration'] = df['Body_Temp'] / (df['Duration'] + eps)
    df['HR_per_Duration'] = df['Heart_Rate'] / (df['Duration'] + eps)

    # BMI & BSA
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2 + eps)
    df['BSA'] = 0.007184 * (df['Height'] ** 0.725) * (df['Weight'] ** 0.425)

    # 심박수 여유 (Heart Rate Reserve 추정)
    df['HR_Reserve'] = 220 - df.get('Age', 40) - df['Heart_Rate']
    df['Heart_Rate_per_Age'] = df['Heart_Rate'] / (df.get('Age', 40) + eps)

    # 운동 강도 지표 유사 (MET, TRIMP 등 단순화 버전)
    df['MET_approx'] = df['Heart_Rate'] * df['Duration'] / 1000
    df['TRIMP_approx'] = df['HR_Duration'] * df['BT_Duration'] / 1000

    # 다항 특성 (비선형 고려)
    df['Duration_squared'] = df['Duration'] ** 2
    df['BMI_squared'] = df['BMI'] ** 2
    df['BMI_cubed'] = df['BMI'] ** 3

    # 범주형 변수 처리 (예: Sex)
    if 'Sex' in df.columns:
        df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1}).fillna(0)

    return df
