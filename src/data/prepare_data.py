import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 상위 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.database import get_database_connection, fetch_air_quality_data

def preprocess_data(df):
    df = df.copy()

    # 🔍 measure_date 타입 확인 및 변환
    print("🔍 measure_date 원본 예시:", df['measure_date'].head())
    if not pd.api.types.is_datetime64_any_dtype(df['measure_date']):
        if pd.api.types.is_numeric_dtype(df['measure_date']):
            df['measure_date'] = df['measure_date'].astype(str).str.zfill(14)
            df['measure_date'] = pd.to_datetime(df['measure_date'], format="%Y%m%d%H%M%S", errors='coerce')
        elif pd.api.types.is_string_dtype(df['measure_date']):
            df['measure_date'] = df['measure_date'].str.zfill(14)
            df['measure_date'] = pd.to_datetime(df['measure_date'], format="%Y%m%d%H%M%S", errors='coerce')

    # 변환 확인 및 제거
    print("🕵️‍♀️ measure_date 변환 실패 건수:", df['measure_date'].isna().sum())
    df = df[df['measure_date'].notna()].copy()
    print("🧹 유효한 measure_date 남은 건수:", len(df))
    print("📌 measure_date dtype:", df['measure_date'].dtype)
    print("📌 measure_date 예시:", df['measure_date'].head(1))

    # 수치형 컬럼 처리
    measure_cols = [col for col in df.columns if 'measure' in col or 'stdr' in col]
    for col in measure_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 완전 결측 컬럼 제거
    df.dropna(axis=1, how='all', inplace=True)
    print("📌 남은 컬럼:", df.columns.tolist())

    # 결측 보간
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # 이상치 처리
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # 시계열 파생
    df['hour'] = df['measure_date'].dt.hour
    df['day_of_week'] = df['measure_date'].dt.dayofweek
    df['month'] = df['measure_date'].dt.month

    return df

def prepare_time_series_data(df, target_columns=['nox_measure', 'sox_measure', 'tsp_measure']):
    stdr_cols = [col for col in df.columns if 'stdr' in col and pd.api.types.is_numeric_dtype(df[col])]
    X = df[['hour', 'day_of_week', 'month'] + stdr_cols]
    y = df[target_columns]
    return X, y

def main():
    connection = get_database_connection()
    if connection is None:
        print("데이터베이스 연결 실패")
        return

    try:
        df = fetch_air_quality_data(connection)
        if df is None:
            print("데이터 조회 실패")
            return

        print("✅ 데이터베이스 연결 성공")
        print("📊 불러온 데이터 수:", len(df))
        print("🧾 컬럼명 확인:", df.columns.tolist())

        df_processed = preprocess_data(df)

        os.makedirs('data/processed', exist_ok=True)
        df_processed.to_csv('data/processed/air_quality_processed.csv', index=False)
        print("✅ 데이터 전처리 완료")

        X, y = prepare_time_series_data(df_processed)
        X.to_csv('data/processed/features.csv', index=False)
        y.to_csv('data/processed/targets.csv', index=False)
        print("✅ 시계열 데이터 준비 완료")

    finally:
        connection.close()
        print("🔌 데이터베이스 연결 종료")

if __name__ == "__main__":
    main()
