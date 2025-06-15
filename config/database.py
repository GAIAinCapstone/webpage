import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
import pandas as pd

# 환경 변수 로드
load_dotenv()
DEFAULT_DB = os.getenv("DEFAULT_DB", "cleansys")

def get_database_connection(database_name=None):
    """
    MySQL 데이터베이스 연결을 생성하고 반환합니다.

    Args:
        database_name (str): 사용할 데이터베이스 이름. 기본값은 .env 또는 'cleansys'.

    Returns:
        MySQL 커넥션 객체
    """
    db_name = database_name or DEFAULT_DB
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='ksm',
            port=3307,
            password='capstone',
            database=db_name
        )
        if connection.is_connected():
            print(f"✅ DB({db_name}) 연결 성공")
            return connection
    except Error as e:
        print(f"❌ DB 연결 오류: {e}")
        return None

def fetch_pollutant_data(table_name, year=None, database_name=None):
    conn = get_database_connection(database_name)
    try:
        with conn.cursor(dictionary=True) as cursor:
            query = f"SELECT * FROM `{table_name}`"
            if year:
                query += " WHERE 정보일시 BETWEEN %s AND %s"
                start = f"{year}-01-01 00:00:00"
                end = f"{year}-12-31 23:59:59"
                cursor.execute(query, (start, end))
            else:
                cursor.execute(query)
            return pd.DataFrame(cursor.fetchall())
    finally:
        conn.close()

def fetch_pollutant_data_multi(table_names, year, database_name):
    """
    여러 테이블에서 동일한 기간의 데이터를 가져와 병합합니다.

    Args:
        table_names (list[str]): 테이블 이름 목록 (예: ["Jugyomyeon2023", "BoryeongPort2023"])
        year (int): 조회할 연도 (예: 2023)
        database_name (str): 연결할 DB 이름 (예: "airKorea")

    Returns:
        pd.DataFrame: 병합된 데이터프레임 (출처별 라벨 포함)
    """
    conn = get_database_connection(database_name)
    all_data = []
    try:
        with conn.cursor(dictionary=True) as cursor:
            start = f"{year}-01-01 01:00:00"
            end = f"{year+1}-12-31 00:00:00"
            for table in table_names:
                query = f"""
                    SELECT * FROM `{table}`
                    WHERE measure_date BETWEEN %s AND %s
                """
                try:
                    cursor.execute(query, (start, end))
                    df = pd.DataFrame(cursor.fetchall())
                    if not df.empty:
                        df["출처"] = table  # 어느 테이블에서 왔는지 표시
                        all_data.append(df)
                except Exception as e:
                    print(f"⚠️ 테이블 {table} 조회 실패: {e}")
        return pd.concat(all_data) if all_data else pd.DataFrame()
    finally:
        conn.close()

def fetch_air_quality_data(connection, start_date=None, end_date=None):
    """
    대기질 데이터를 가져옵니다.
    """
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT 
            measure_date, fact_name, area_nm, stack_code,
            nox_measure, nox_stdr, sox_measure, sox_stdr,
            tsp_measure, tsp_stdr, nh3_measure, nh3_stdt,
            hf_measure, hf_stdr, hcl_measure, hcl_stdr,
            co_measure, co_stdr
        FROM api_data
        """
        if start_date and end_date:
            query += " WHERE measure_date BETWEEN %s AND %s"
            cursor.execute(query, (start_date, end_date))
        else:
            cursor.execute(query)
        return pd.DataFrame(cursor.fetchall())
    except Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()

def get_factory_list(connection):
    """공장 목록을 가져옵니다."""
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT fact_name FROM api_data")
        return [row[0] for row in cursor.fetchall()]
    except Error as e:
        print(f"공장 목록 조회 중 오류 발생: {e}")
        return []
    finally:
        if cursor:
            cursor.close()

def get_area_list(connection):
    """지역 목록을 가져옵니다."""
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT area_nm FROM api_data")
        return [row[0] for row in cursor.fetchall()]
    except Error as e:
        print(f"지역 목록 조회 중 오류 발생: {e}")
        return []
    finally:
        if cursor:
            cursor.close()