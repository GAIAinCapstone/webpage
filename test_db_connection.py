from config.database import get_database_connection, fetch_air_quality_data

def test_connection():
    # 데이터베이스 연결
    print("데이터베이스 연결 시도 중...")
    connection = get_database_connection()
    
    if connection is None:
        print("데이터베이스 연결 실패")
        return
    
    try:
        # 데이터 가져오기 테스트
        print("\n데이터 조회 시도 중...")
        df = fetch_air_quality_data(connection)
        
        if df is not None:
            print("\n데이터 조회 성공!")
            print("\n데이터 샘플:")
            print(df.head())
            print("\n데이터 크기:", df.shape)
            print("\n컬럼 목록:", df.columns.tolist())
        else:
            print("데이터 조회 실패")
            
    finally:
        connection.close()
        print("\n데이터베이스 연결 종료")

if __name__ == "__main__":
    test_connection() 