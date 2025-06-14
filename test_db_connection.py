from config.database import get_database_connection, fetch_air_quality_data

def test_connection(database_name='cleansys'):
    """
    주어진 데이터베이스에 연결하여 대기질 데이터를 가져오는 테스트 함수.
    
    Args:
        database_name (str): 연결할 데이터베이스 이름 ('cleansys' 또는 'weatherCenter')
    """
    print(f"데이터베이스 '{database_name}' 연결 시도 중...")
    connection = get_database_connection(database=database_name)
    
    if connection is None:
        print("❌ 데이터베이스 연결 실패")
        return
    
    try:
        print("\n✅ 데이터 조회 시도 중...")
        df = fetch_air_quality_data(connection)
        
        if df is not None:
            print("✅ 데이터 조회 성공!")
            print("\n📊 데이터 샘플:")
            print(df.head())
            print(f"\n🔢 데이터 크기: {df.shape}")
            print(f"🧾 컬럼 목록: {df.columns.tolist()}")
        else:
            print("❌ 데이터 조회 실패")
            
    finally:
        connection.close()
        print("\n🔌 데이터베이스 연결 종료")

if __name__ == "__main__":
    # 'cleansys' 또는 'weatherCenter' 등으로 바꿔서 테스트 가능
    test_connection('cleansys')
