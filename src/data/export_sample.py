# src/data/export_sample.py
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.database import get_database_connection, fetch_air_quality_data

def main():
    conn = get_database_connection()
    if conn is None:
        print("❌ DB 연결 실패")
        return

    try:
        df = fetch_air_quality_data(conn)
        print("✅ 데이터 수:", len(df))
        print("📌 컬럼 목록:", df.columns.tolist())

        # 상위 30개만 저장
        sample_df = df.head(30)
        os.makedirs('data/sample', exist_ok=True)
        sample_df.to_csv("data/sample/raw_sample.csv", index=False)
        print("✅ 샘플 저장 완료: data/sample/raw_sample.csv")

    finally:
        conn.close()
        print("🔌 연결 종료")

if __name__ == "__main__":
    main()
