# 📁 config/database.py

import os
import pymysql
import pandas as pd
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_database_connection():
    """
    MySQL 데이터베이스 연결을 생성하고 반환합니다.
    """
    try:
        connection = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", 56796)),
            user=os.getenv("MYSQL_USER", "ksw"),
            password=os.getenv("MYSQL_PASSWORD", "capstone"),
            db=os.getenv("MYSQL_DB", "weatherCenter"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("✅ 데이터베이스 연결 성공")
        return connection
    except Exception as e:
        print(f"❌ 데이터베이스 연결 중 오류 발생: {e}")
        return None

def fetch_air_quality_data(connection, start_date=None, end_date=None):
    """
    대기질 데이터를 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
        start_date: 시작 날짜 (선택사항)
        end_date: 종료 날짜 (선택사항)
    
    Returns:
        DataFrame: 대기질 데이터
    """
    try:
        with connection.cursor() as cursor:
            query = """
            SELECT 
                measure_date,
                fact_name,
                area_nm,
                stack_code,
                nox_measure,
                nox_stdr,
                sox_measure,
                sox_stdr,
                tsp_measure,
                tsp_stdr,
                nh3_measure,
                nh3_stdr,
                hf_measure,
                hf_stdr,
                hcl_measure,
                hcl_stdr,
                co_measure,
                co_stdr
            FROM api_data
            """
            
            if start_date and end_date:
                query += " WHERE measure_date BETWEEN %s AND %s"
                cursor.execute(query, (start_date, end_date))
            else:
                cursor.execute(query)
                
            data = cursor.fetchall()
            return pd.DataFrame(data)
    except Exception as e:
        print(f"❌ 데이터 조회 중 오류 발생: {e}")
        return None

def get_factory_list(connection):
    """
    공장 목록을 가져옵니다.
    
    Returns:
        list: 공장 목록
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT fact_name FROM api_data")
            return [row['fact_name'] for row in cursor.fetchall()]
    except Exception as e:
        print(f"❌ 공장 목록 조회 중 오류 발생: {e}")
        return []

def get_area_list(connection):
    """
    지역 목록을 가져옵니다.
    
    Returns:
        list: 지역 목록
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT area_nm FROM api_data")
            return [row['area_nm'] for row in cursor.fetchall()]
    except Exception as e:
        print(f"❌ 지역 목록 조회 중 오류 발생: {e}")
        return []
