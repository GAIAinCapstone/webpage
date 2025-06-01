from config.database import get_database_connection, fetch_air_quality_data

def main():
    # DB 연결
    connection = get_database_connection()
    if connection is None:
        print("❌ DB 연결 실패")
        return

    try:
        # 데이터 불러오기
        df = fetch_air_quality_data(connection)

        print("✅ 데이터프레임 구조:")
        print(df.info())
        print("\n📊 수치 요약 (describe):")
        print(df.describe())
        print("\n🕳️ 결측치 개수:")
        print(df.isna().sum())
        print("\n🏭 공장 이름별 샘플 수 (fact_name):")
        print(df['fact_name'].value_counts())
    finally:
        connection.close()

if __name__ == "__main__":
    main()
