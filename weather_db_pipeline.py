import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
from src.models.diffusion import DiffusionCoefficient
from datetime import datetime

def get_weathercenter_connection():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='ksm',
            password='capstone',
            database='weatherCenter',
            port=3307
        )
        if connection.is_connected():
            print("✅ weatherCenter 연결 성공")
            return connection
    except Error as e:
        print(f"❌ 연결 실패: {e}")
        return None

def fetch_wind_weather(year):
    conn = get_weathercenter_connection()
    try:
        wind_query = f"SELECT * FROM about_wind{year}"
        weather_query = f"SELECT * FROM weather{year}"
        wind = pd.read_sql(wind_query, conn)
        weather = pd.read_sql(weather_query, conn)
        return wind, weather
    finally:
        conn.close()

def preprocess_and_merge(wind, weather):
    wind.columns = ['region', 'datetime', 'speed', 'direction']
    weather.columns = [
        'region', 'datetime', 'temperature', 'rain', 'humidity',
        'landPressure', 'seaPressure', 'sun_sa', 'sun_jo',
        'total_cloud', 'lowmiddle_cloud'
    ]

    wind['datetime'] = pd.to_datetime(wind['datetime'])
    weather['datetime'] = pd.to_datetime(weather['datetime'])

    df = pd.merge(wind, weather, on=['region', 'datetime'], how='outer')
    df = df.sort_values('datetime')

    df = df.set_index('datetime')
    df.interpolate(method='time', inplace=True)
    df = df.reset_index()

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    df['hour'] = df['datetime'].dt.hour
    df['is_daytime'] = df['hour'].between(6, 18)

    diff = DiffusionCoefficient()
    
    def calc_stability(row):
        try:
            if row['is_daytime']:
                cond = diff.classify_insolation(row['sun_sa'])
                return diff.get_stability(row['speed'], cond, True)
            else:
                cond = diff.classify_cloudiness(row['total_cloud'])
                return diff.get_stability(row['speed'], cond, False)
        except:
            return 'C'

    df['stability'] = df.apply(calc_stability, axis=1)
    return df[['datetime', 'region', 'speed', 'direction', 'temperature', 'humidity', 'sun_sa', 'total_cloud', 'stability']]

def upload_to_db(df, table_name):
    conn = get_weathercenter_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"""
            CREATE TABLE {table_name} (
                datetime DATETIME,
                region VARCHAR(20),
                speed FLOAT,
                direction FLOAT,
                temperature FLOAT,
                humidity FLOAT,
                sun_sa FLOAT,
                total_cloud FLOAT,
                stability CHAR(1)
            )
        """)
        conn.commit()

        for _, row in df.iterrows():
            cursor.execute(f"""
                INSERT INTO {table_name} (datetime, region, speed, direction, temperature, humidity, sun_sa, total_cloud, stability)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))
        conn.commit()
        print(f"✅ {table_name} 테이블에 {len(df)}행 업로드 완료")
    except Error as e:
        print(f"❌ 업로드 실패: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    for year in range(2018, 2025):
        print(f"--- {year} 데이터 처리 중 ---")
        wind, weather = fetch_wind_weather(year)
        processed = preprocess_and_merge(wind, weather)
        upload_to_db(processed, f"processed_weather_{year}")
