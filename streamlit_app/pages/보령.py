# streamlit_app/pages/보령.py
import streamlit as st
from config.database import fetch_pollutant_data
import pandas as pd

st.title("보령 발전소 오염물질 시각화")

pollutant = st.selectbox("오염물질 선택", ["NOx", "SOx", "TSP", "O₂", "FL1", "TMP"])
year = st.selectbox("연도 선택", list(range(2018, 2026)))

# 테이블명 매핑
table_map = {
    "NOx": "tms_보령_nox",
    "SOx": "tms_보령_sox",
    "TSP": "tms_보령_tsp",
    "O₂":  "tms_보령_o₂",
    "FL1": "tms_보령_fl1",
    "TMP": "tms_보령_tmp",
}

table_name = table_map[pollutant]
df = fetch_pollutant_data(table_name, year)

st.markdown(f"### {pollutant} 측정값 ({year}년)")
if not df.empty:
    df["정보일시"] = pd.to_datetime(df["정보일시"])
    df = df.set_index("정보일시")
    st.line_chart(df["값"])
    st.dataframe(df.head(100))
else:
    st.warning("해당 연도에 데이터가 없습니다.")
