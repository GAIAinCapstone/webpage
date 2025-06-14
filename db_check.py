from config.database import get_database_connection
import pandas as pd

def inspect_table(db_name, table_name, limit=5):
    conn = get_database_connection(db_name)
    try:
        query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        df = pd.read_sql(query, conn)
        print(f"\n📄 {db_name}.{table_name} 미리보기 (상위 {limit}행):\n")
        print(df.head())
        print("\n🔍 컬럼 정보 (컬럼명 + 타입):\n")
        print(df.dtypes)
    finally:
        conn.close()

# 예시: 주요 테이블 3개 확인
inspect_table("weatherCenter", "processed_weather_2023")
inspect_table("weatherCenter", "about_wind2023")
inspect_table("weatherCenter", "weather2023")
inspect_table("airKorea", "BoryeongPort23")  # 정확한 테이블명으로 바꿔줘
inspect_table("cleansys", "tms_보령_nox")  # 정확한 테이블명으로 바꿔줘
inspect_table("cleansys", "tms_신보령_sox")  # 정확한 테이블명으로 바꿔줘
inspect_table("cleansys", "tms_신서천_tsp")  # 정확한 테이블명으로 바꿔줘
