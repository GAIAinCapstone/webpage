import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from datetime import timedelta
from config.database import get_database_connection, fetch_air_quality_data

FEATURE_COLUMNS = ['speed', 'direction', 'temperature', 'humidity', 'sun_sa', 'total_cloud']
TARGET_COLUMNS = ['nox_measure', 'sox_measure', 'tsp_measure']

def load_weather_data(year):
    conn = get_database_connection(database_name='weatherCenter')
    query = f"SELECT * FROM processed_weather_{year}"
    df = pd.read_sql(query, conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def load_target_data(start, end):
    conn = get_database_connection(database_name='cleansys')
    df = fetch_air_quality_data(conn, start, end)
    conn.close()
    df = df[['measure_date', 'nox_measure', 'sox_measure', 'tsp_measure']]
    df = df.rename(columns={'measure_date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def generate_feature_target_df(weather_df, pollutant_df):
    # inner join by datetime
    merged = pd.merge(weather_df, pollutant_df, on='datetime', how='inner')

    features = merged[FEATURE_COLUMNS]
    targets = merged[TARGET_COLUMNS]

    features.to_csv('data/processed/features.csv', index=False)
    targets.to_csv('data/processed/targets.csv', index=False)
    print("✅ features.csv, targets.csv 저장 완료!")

if __name__ == "__main__":
    year = 2023
    weather_df = load_weather_data(year)
    pollutant_df = load_target_data(f"{year}-01-01", f"{year}-12-31")
    generate_feature_target_df(weather_df, pollutant_df)
