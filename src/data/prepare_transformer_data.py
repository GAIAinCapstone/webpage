import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# 🔧 config 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.database import fetch_pollutant_data, get_database_connection

def load_all_weather_data(start_year=2018, end_year=2024):
    conn = get_database_connection(database_name='weatherCenter')
    all_weather = []
    try:
        for year in range(start_year, end_year + 1):
            table_name = f"processed_weather_{year}"
            query = f"SELECT * FROM {table_name} WHERE region = '보령'"
            df = pd.read_sql(query, conn)
            df['datetime'] = pd.to_datetime(df['datetime'])
            all_weather.append(df)
    finally:
        conn.close()
    return pd.concat(all_weather)

def load_all_pollutant_data():
    stations = ['보령', '신보령', '신서천']
    pollutants = ['nox', 'sox', 'tsp']
    all_data = []
    for station in stations:
        for pollutant in pollutants:
            table_name = f"tms_{station}_{pollutant}"
            df = fetch_pollutant_data(table_name, database_name='cleansys')
            if df.empty:
                continue
            df['정보일시'] = pd.to_datetime(df['정보일시'])
            df = df.rename(columns={'정보일시': 'datetime', '값': f'{station}_{pollutant}'})
            all_data.append(df.set_index('datetime'))
    return pd.concat(all_data, axis=1).reset_index()

# 데이터 로딩
weather_df = load_all_weather_data()
pollutant_df = load_all_pollutant_data()

# 시간 단위로 정렬 및 병합
weather_df['datetime_rounded'] = weather_df['datetime'].dt.floor('H')
pollutant_df['datetime_rounded'] = pollutant_df['datetime'].dt.floor('H')
merged_df = pd.merge(weather_df, pollutant_df, on='datetime_rounded', how='inner')

# 특성과 타깃 선택
feature_cols = ['speed', 'direction', 'temperature', 'humidity', 'sun_sa', 'total_cloud']
target_cols = [col for col in merged_df.columns if any(pol in col for pol in ['nox', 'sox', 'tsp'])]

features = merged_df[feature_cols].dropna()
targets = merged_df[target_cols].dropna()

min_len = min(len(features), len(targets))
features = features.iloc[:min_len]
targets = targets.iloc[:min_len]

os.makedirs("data/processed", exist_ok=True)
features.to_csv("data/processed/features.csv", index=False)
targets.to_csv("data/processed/targets.csv", index=False)

print("✅ features.csv / targets.csv 생성 완료!")
