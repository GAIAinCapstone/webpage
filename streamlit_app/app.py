import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import get_database_connection, fetch_pollutant_data
from src.models.aermod_simulator import GaussianPlumeModel
from src.models.transformer_model import SimpleTransformer
from src.models.diffusion import DiffusionCoefficient

st.set_page_config(layout="wide")
st.title("화력발전소 → 대기오염 확산 분석")

# ------------------------- Sidebar -------------------------
st.sidebar.header("AI 기반 대기오염 예측 시스템")
plant = st.sidebar.selectbox("화력발전소 선택", ["보령", "신보령", "신서천"])
pollutant = st.sidebar.selectbox("오염물질 선택", ["NOx", "SOx", "TSP"])
year = st.sidebar.selectbox("기상 데이터 연도", list(range(2018, 2025)))
station = st.sidebar.selectbox("분석 관찰 (기상 기준)", ["BoryeongPort24", "Jugyomyeon24", "Seomyeon24"])
Q = st.sidebar.slider("발추량 (g/s)", 0.0, 100.0, 10.0)

if st.sidebar.button("분석 실행"):
    st.session_state['run'] = True

# ------------------------- Load Data Functions -------------------------
def load_processed_weather(year):
    conn = get_database_connection("weatherCenter")
    query = f"SELECT * FROM processed_weather_{year}"
    df = pd.read_sql(query, conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values("datetime")

def load_airkorea_data(station, year):
    conn = get_database_connection("airKorea")
    query = f"""
    SELECT * FROM `{station}`
    WHERE measure_date BETWEEN '{year}-01-01' AND '{year}-12-31'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['measure_date'] = pd.to_datetime(df['measure_date'])
    return df.sort_values("measure_date")

def load_stack_data(plant, pollutant, year):
    map_name = {"보령": "tms_보령", "신보령": "tms_신보령", "신서천": "tms_신서천"}
    table = f"{map_name[plant]}_{pollutant.lower()}"
    return fetch_pollutant_data(table, year, database_name="cleansys")

# ------------------------- Run Main Logic -------------------------
if 'run' in st.session_state:
    weather = load_processed_weather(year)
    air = load_airkorea_data(station, year)
    stack = load_stack_data(plant, pollutant, year)

    st.subheader(f"{year}년 기상 데이터")
    st.dataframe(weather.head())

    # Diffusion 계산
    diff = DiffusionCoefficient()
    weather['is_daytime'] = weather['datetime'].dt.hour.between(6, 18)
    weather['stability'] = weather.apply(
        lambda row: diff.get_stability(
            row['speed'],
            diff.classify_insolation(row['sun_sa']) if row['is_daytime'] else diff.classify_cloudiness(row['total_cloud']),
            row['is_daytime']), axis=1)
    sigma_y = diff.calculation_y(100.0, weather['stability'].iloc[-1])
    sigma_z = diff.calculation_z(100.0, weather['stability'].iloc[-1])

    # AERMOD 계산
    u_list = weather['speed'].values[:10]
    Q_list = u_list * Q  # 풍속 기준 임의 설정
    H, x, y, z = 50.0, 100.0, 0.0, 0.0
    aermod_results = [GaussianPlumeModel(Q_, u_, H, sigma_y, sigma_z).concentration(x, y, z)
                      for Q_, u_ in zip(Q_list, u_list)]

    # Transformer 예측
    features = weather[['speed', 'direction', 'temperature', 'humidity', 'sun_sa', 'total_cloud']].tail(24).values.astype(np.float32)
    model_tr = SimpleTransformer(features.shape[1], 3)
    try:
        model_tr.load_state_dict(torch.load('src/models/transformer_model.pt', map_location='cpu'))
        model_tr.eval()
        with torch.no_grad():
            input_seq = torch.tensor(features).unsqueeze(0)
            pred = model_tr(input_seq).numpy().flatten()
    except Exception as e:
        st.warning(f"Transformer 모델 오류: {e}")
        pred = [np.nan, np.nan, np.nan]

    # 지도 시각화용
    np.random.seed(0)
    lat_center, lon_center = 36.349, 126.604
    lats = lat_center + 0.01 * (np.random.rand(10) - 0.5)
    lons = lon_center + 0.01 * (np.random.rand(10) - 0.5)
    map_df = pd.DataFrame({"lat": lats, "lon": lons, "농도": aermod_results})

    st.subheader("🌎 지도 시각화 (AERMOD 결과)")
    fig = px.scatter_mapbox(map_df, lat='lat', lon='lon', color='농도',
                            color_continuous_scale='Jet', size='농도', size_max=20,
                            zoom=10, height=400, mapbox_style='carto-positron')
    st.plotly_chart(fig, use_container_width=True)

    # 실측 비교
    pollutant_col = f"{pollutant.upper()}_value" if pollutant != "NOx" else "NO2_value"
    if pollutant_col in air.columns:
        st.subheader(f"실측 측정소 데이터 vs 예측")
        air['time'] = air['measure_date']
        line_df = pd.DataFrame({
            '측정소 실측': air[pollutant_col].values[:10],
            'AERMOD 예측': aermod_results[:10],
        }, index=air['time'].values[:10])
        st.line_chart(line_df)

    st.subheader("Transformer 시계열 예측 결과")
    st.write(f"NOx 예측: {pred[0]:.2f}, SOx 예측: {pred[1]:.2f}, TSP 예측: {pred[2]:.2f}")
