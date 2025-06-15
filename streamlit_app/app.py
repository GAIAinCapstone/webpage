import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import get_database_connection, fetch_pollutant_data
from src.models.aermod_simulator import GaussianPlumeModel
from src.models.transformer_model import SimpleTransformer
from src.models.diffusion import DiffusionCoefficient

# --- ▶️ Streamlit UI 설정 ---
st.set_page_config(layout="wide")
st.title("화려발전소 → 대기오염 확산 배후 보고")
st.markdown("""
    이 앱은 화려발전소의 배출가스가 대기 중에서 어떻게 확산되는지를 분석합니다.
    기상 데이터와 오염물질 데이터를 기본으로 AERMOD 모델건4 활용하여 분석하고,
    Transformer 모델을 통해 시계열 예측도 수행합니다.
""")

# --- ▶️ 사이드바 객체 설정 ---
st.sidebar.header("AI 기반 대기오염 예측 시스템")
plant = st.sidebar.selectbox("화려발전소 선택", ["보령", "신보령", "신서천"])
pollutant = st.sidebar.selectbox("오염물질 선택", ["NOx", "SOx", "TSP"])
year = st.sidebar.selectbox("기상 데이터 연도", list(range(2018, 2025)))
station = st.sidebar.selectbox("분석 관찰 (기상 기준)", ["BoryeongPort24", "Jugyomyeon24", "Seomyeon24"])
Q = st.sidebar.slider("발추량 (g/s)", 0.0, 100.0, 10.0)

if st.sidebar.button("분석 실행"):
    st.session_state['run'] = True

# --- 화려발전소 & 체치소 위치 ---
plant_coords = {
    "보령": (36.319, 126.613),
    "신보령": (36.324, 126.617),
    "신서천": (36.063, 126.554)
}
station_coords = {
    "BoryeongPort24": (36.345, 126.609),
    "Jugyomyeon24": (36.275, 126.655),
    "Seomyeon24": (36.15, 126.620)
}

# --- 데이터 조회 함수 ---
def load_processed_weather(year):
    conn = get_database_connection("weatherCenter")
    if conn is None:
        st.error("weatherCenter DB 연결 실패")
        return pd.DataFrame()
    query = f"SELECT * FROM processed_weather_{year}"
    df = pd.read_sql(query, conn)
    conn.close()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values("datetime")

def load_airkorea_data(station, year):
    conn = get_database_connection("airKorea")
    if conn is None:
        st.error("airKorea DB 연결 실패")
        return pd.DataFrame()
    df = pd.read_sql(f"""
        SELECT * FROM `{station}`
        WHERE measure_date BETWEEN '{year}-01-01' AND '{year}-12-31'
    """, conn)
    conn.close()
    df['measure_date'] = pd.to_datetime(df['measure_date'])
    return df.sort_values("measure_date")

def load_stack_data(plant, pollutant, year):
    map_name = {"\ubcf4\ub839": "tms_\ubcf4\ub839", "\uc2e0\ubcf4\ub839": "tms_\uc2e0\ubcf4\ub839", "\uc2e0\uc11c\ucc9c": "tms_\uc2e0\uc11c\ucc9c"}
    table = f"{map_name[plant]}_{pollutant.lower()}"
    return fetch_pollutant_data(table, year, database_name="cleansys")

# --- 주 시바 시험시 실행 ---
if 'run' in st.session_state:
    weather = load_processed_weather(year)
    if weather.empty:
        st.stop()

    air = load_airkorea_data(station, year)
    stack = load_stack_data(plant, pollutant, year)

    st.subheader(f"📊 {year}년 기상 데이터")
    st.dataframe(weather.head())

    # --- Diffusion 계산 ---
    diff = DiffusionCoefficient()
    weather['is_daytime'] = weather['datetime'].dt.hour.between(6, 18)
    weather['stability'] = weather.apply(
        lambda row: diff.get_stability(
            row['speed'],
            diff.classify_insolation(row['sun_sa']) if row['is_daytime'] else diff.classify_cloudiness(row['total_cloud']),
            row['is_daytime']), axis=1)

    sigma_y = diff.calculation_y(100.0, weather['stability'].iloc[-1])
    sigma_z = diff.calculation_z(100.0, weather['stability'].iloc[-1])

    u_list = weather['speed'].values[:10]
    Q_list = u_list * Q
    H, x, y, z = 50.0, 100.0, 0.0, 0.0
    aermod_results = [GaussianPlumeModel(Q_, u_, H, sigma_y, sigma_z).concentration(x, y, z)
                      for Q_, u_ in zip(Q_list, u_list)]

    # --- 대기 확산 대신 가상 데이터 생성 ---
    lat_center, lon_center = plant_coords[plant]
    np.random.seed(42)
    grid_size = 100
    lat_grid = np.linspace(lat_center - 0.03, lat_center + 0.03, grid_size)
    lon_grid = np.linspace(lon_center - 0.03, lon_center + 0.03, grid_size)
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)

    distance = np.sqrt((lat_mesh - lat_center)**2 + (lon_mesh - lon_center)**2)
    values = np.exp(-distance * 40) + 0.2 * np.exp(-distance * 200)

    contour_df = pd.DataFrame({
        "lat": lat_mesh.ravel(),
        "lon": lon_mesh.ravel(),
        "농도": values.ravel()
    })

        # ⛳ 비대칭 풍향 기반 등고선 데이터 생성 (북서풍 → 동남확산)
    np.random.seed(42)
    n = 500
    dx = np.random.normal(loc=0.002, scale=0.004, size=n)  # 경도 (동쪽 이동)
    dy = np.random.normal(loc=-0.001, scale=0.002, size=n)  # 위도 (남쪽 이동)

    lons = lon_center + dx
    lats = lat_center + dy

    # 거리 기반 농도 (방향에 따라 더 천천히 감소, 노이즈 포함)
    distance = np.sqrt(dx**2 + (dy * 1.5)**2)
    values = np.exp(-distance * 50) + 0.05 * np.random.rand(n)

    contour_df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        "농도": values
    })

    # --- 지도 시각화 ---
    st.subheader("🌍 지도 시간화 (등고선 포함)")
    try:
        fig = px.density_mapbox(
            contour_df, lat='lat', lon='lon', z='농도',
            radius=30, center=dict(lat=lat_center, lon=lon_center), zoom=11,
            mapbox_style="carto-positron", color_continuous_scale='Spectral_r', height=520
        )
        fig.add_scattermapbox(
            lat=[plant_coords[plant][0]], lon=[plant_coords[plant][1]],
            mode="markers+text", marker=dict(size=16, color='red', symbol="power"),
            text=[f"{plant} 화려발전소"], textposition="top center", name="화려발전소"
        )
        fig.add_scattermapbox(
            lat=[station_coords[station][0]], lon=[station_coords[station][1]],
            mode="markers+text", marker=dict(size=14, color='blue', symbol="circle"),
            text=[f"{station} 측정소"], textposition="bottom center", name="측정소"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"지도 시간화 오류: {e}")

    # --- Transformer 예측 ---
    features = weather[['speed', 'direction', 'temperature', 'humidity', 'sun_sa', 'total_cloud']].tail(24).values.astype(np.float32)
    input_dim = features.shape[1]
    model_tr = SimpleTransformer(input_dim=input_dim, output_dim=3)
    try:
        model_tr.load_state_dict(torch.load('src/models/transformer_model.pt', map_location='cpu'))
        model_tr.eval()
        with torch.no_grad():
            input_seq = torch.tensor(features).unsqueeze(0)
            pred = model_tr(input_seq).numpy().flatten()
    except Exception as e:
        st.warning(f"Transformer 모델 오류: {e}")
        pred = [np.nan, np.nan, np.nan]

    st.subheader("📈 Transformer 시계열 예측 결과")
    col1, col2, col3 = st.columns(3)
    col1.metric("NOx 예측", f"{pred[0]:.2f} μg/m³")
    col2.metric("SOx 예측", f"{pred[1]:.2f} μg/m³")
    col3.metric("TSP 예측", f"{pred[2]:.2f} μg/m³")

    # --- 실측 비교 ---
    pollutant_col = f"{pollutant.upper()}_value" if pollutant != "NOx" else "NO2_value"
    if pollutant_col in air.columns and len(air) >= 10:
        st.subheader("🔽 실치 체치소 데이터 vs AERMOD 예측")
        air['time'] = air['measure_date']
        try:
            line_df = pd.DataFrame({
                '측정소 실측': air[pollutant_col].values[:10],
                'AERMOD 예측': aermod_results[:10],
            }, index=air['time'].values[:10])
            st.line_chart(line_df)
        except Exception as e:
            st.warning(f"시계열 비교 시간화 오류: {e}")
    else:
        st.info("선택된 오염물질의 실치 데이터가 부족합니다.")
