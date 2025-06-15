import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config.database import fetch_pollutant_data, fetch_pollutant_data_multi
import pandas as pd
import altair as alt


st.title("신보령 발전소 오염물질 시각화")

pollutant = st.selectbox("오염물질 선택", ["NOx", "SOx", "TSP", "O₂", "FL1", "TMP"])
year = st.selectbox("연도 선택", list(range(2018, 2025)))

# 테이블명 매핑
cleansys_table_map = {
    "NOx": "tms_신보령_nox",
    "SOx": "tms_신보령_sox",
    "TSP": "tms_신보령_tsp",
    "O₂":  "tms_신보령_o₂",
    "FL1": "tms_신보령_fl1",
    "TMP": "tms_신보령_tmp",
}
cleansys_table_name = cleansys_table_map[pollutant]
cleansys = fetch_pollutant_data(cleansys_table_name, year)
cleansys["출처"] = "신보령발전소"

factory_table_map = [f"Jugyomyeon{str(year)[2:]}"]
if str(year)[2:] in [str(23), str(24)]:
    print("BoryeongPort 테이블을 추가합니다.")
    factory_table_map.append(f"BoryeongPort{str(year)[2:]}")
factory = fetch_pollutant_data_multi(factory_table_map, year, "airKorea")

if not cleansys.empty:
    cleansys["정보일시"] = pd.to_datetime(cleansys["정보일시"])
    cleansys = cleansys.rename(columns={"정보일시": "datetime", "값": "농도"})
    cleansys = cleansys[["datetime", "농도", "출처"]]

if not factory.empty:
    factory["measure_date"] = pd.to_datetime(factory["measure_date"])
    if pollutant == "O₂":
        value_col = "O2_value"
    elif pollutant == "NOx":
        value_col = "NO2_value"
    elif pollutant == "SOx":
        value_col = "SO2_value"
    elif pollutant == "TSP":
        value_col = "PM10_value"
    factory = factory[["measure_date", value_col, "출처"]]
    factory = factory.rename(columns={"measure_date": "datetime", value_col: "농도"})

if not cleansys.empty and not factory.empty:
    merged_df = pd.concat([cleansys, factory])

    fig = px.line(
        merged_df,
        x="datetime",
        y="농도",
        color="출처",
        title="오염물질 농도 비교 (발전소 vs 측정소)",
        labels={
            "datetime": "측정시각",
            "농도": "농도 (mg/m³)",
            "출처": "데이터 출처"
        }
    )

    fig.update_layout(
        hovermode="x unified",  # 툴팁 정렬
        xaxis_rangeslider_visible=True  # 하단 슬라이더 추가 (선택 사항)
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    show_data = pd.merge(cleansys, factory, on="datetime", how="inner")
    st.dataframe(show_data.head(100), use_container_width=True, height=600)

else:
    st.warning("발전소데이터가 없어 그래프를 그릴 수 없습니다.")