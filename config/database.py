import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
import pandas as pd

# .env 파일 로드
load_dotenv()

def get_database_connection():
    """
    MySQL 데이터베이스 연결을 생성하고 반환합니다.
    """
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',  # SSH 터널링을 위한 로컬호스
            user='ksw',
            password='capstone',
            database='cleansys',
            port=3307
             # SSH 터널링 포트
        )
        if connection.is_connected():
            print("데이터베이스 연결 성공")
            return connection
    except Error as e:
        print(f"데이터베이스 연결 중 오류 발생: {e}")
        return None

def fetch_pollutant_data(table_name, year=None):
    conn = get_database_connection()
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
        cursor = connection.cursor(dictionary=True)
        
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
            nh3_stdt,
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
        
    except Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return None
    finally:
        if cursor:
            cursor.close()

def get_factory_list(connection):
    """
    공장 목록을 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
    
    Returns:
        list: 공장 목록
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT fact_name FROM api_data")
        factories = [row[0] for row in cursor.fetchall()]
        return factories
    except Error as e:
        print(f"공장 목록 조회 중 오류 발생: {e}")
        return []
    finally:
        if cursor:
            cursor.close()

def get_area_list(connection):
    """
    지역 목록을 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
    
    Returns:
        list: 지역 목록
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT area_nm FROM api_data")
        areas = [row[0] for row in cursor.fetchall()]
        return areas
    except Error as e:
        print(f"지역 목록 조회 중 오류 발생: {e}")
        return []
    finally:
        if cursor:
            cursor.close() 