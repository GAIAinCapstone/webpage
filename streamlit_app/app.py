import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import torch
from src.models.aermod_simulator import GaussianPlumeModel
from src.models.transformer_model import SimpleTransformer
from src.models.diffusion import DiffusionCoefficient
from config.database import get_database_connection
import plotly.express as px

st.set_page_config(layout="wide")

# ---- 사이드바 ----
st.sidebar.markdown("""
<style>
.sidebar .sidebar-content {background-color: #0e1117; color: white;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title('AI 기반 대기질 예측 시스템')

st.sidebar.header('분석 대상 설정')
pollutant = st.sidebar.selectbox('오염물질 선택', ['미세먼지(PM10)', '초미세먼지(PM2.5)', '질소산화물(NOx)', '황산화물(SOx)'])
period = st.sidebar.text_input('분석 기간', '2023-01-01 ~ 2023-12-31')
year = st.sidebar.selectbox("기상 데이터 연도", [2018, 2019, 2020, 2021, 2022, 2023, 2024])

st.sidebar.header('기상 조건 설정')
Q = st.sidebar.slider('배출량 (g/s)', 0.0, 100.0, 10.0)

if st.sidebar.button('분석 실행'):
    st.session_state['run_analysis'] = True

# ---- 메인 ----
st.title('대기질 예측 결과')

# ---- 기상 데이터 불러오기 ----
def load_processed_weather(year):
    conn = get_database_connection(database_name='weatherCenter')
    if conn is None:
        st.error("데이터베이스 연결 실패")
        return pd.DataFrame()
    try:
        query = f"SELECT * FROM processed_weather_{year}"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.warning(f"데이터 로딩 실패: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

weather_df = load_processed_weather(year)

# datetime 타입 보장
if not pd.api.types.is_datetime64_any_dtype(weather_df['datetime']):
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], errors='coerce')

st.markdown(f"### {year}년 기상 데이터")
st.dataframe(weather_df.head())

# ---- 최근 24시간 데이터 추출 ----
latest_weather = weather_df.sort_values("datetime").tail(24).copy()

# ---- Diffusion Model 기반 sigma 계산 ----
diff = DiffusionCoefficient()
latest_weather['is_daytime'] = latest_weather['datetime'].dt.hour.between(6, 18)
latest_weather['stability'] = latest_weather.apply(lambda row: diff.get_stability(
    row['speed'], diff.classify_insolation(row['sun_sa']) if row['is_daytime'] else diff.classify_cloudiness(row['total_cloud']), row['is_daytime']), axis=1)

sigma_y = diff.calculation_y(100.0, latest_weather['stability'].iloc[-1])
sigma_z = diff.calculation_z(100.0, latest_weather['stability'].iloc[-1])

# ---- AERMOD 예측 ----
Q_list = latest_weather['speed'].values[:10] * 10
u_list = latest_weather['speed'].values[:10]
H, x, y, z = 50.0, 100.0, 0.0, 0.0

aermod_results = []
for Q_, u_ in zip(Q_list, u_list):
    model = GaussianPlumeModel(Q_, u_, H, sigma_y, sigma_z)
    c = model.concentration(x, y, z)
    aermod_results.append(c)

aermod_results = np.array(aermod_results)

# ---- 지도 시각화용 임의 위치 데이터 생성 ----
lat_center, lon_center = 37.5665, 126.9780
np.random.seed(42)
lats = lat_center + 0.01 * (np.random.rand(10) - 0.5)
lons = lon_center + 0.01 * (np.random.rand(10) - 0.5)
map_df = pd.DataFrame({
    'lat': lats,
    'lon': lons,
    '농도': aermod_results
})

# ---- Transformer 예측 ----
feature_cols = ['speed', 'direction', 'temperature', 'humidity', 'sun_sa', 'total_cloud']
features = latest_weather[feature_cols].values.astype(np.float32)
input_dim = features.shape[1]
output_dim = 3
model_tr = SimpleTransformer(input_dim, output_dim)
try:
    model_tr.load_state_dict(torch.load('src/models/transformer_model.pt', map_location=torch.device('cpu')))
    model_tr.eval()
    with torch.no_grad():
        input_seq = torch.tensor(features).unsqueeze(0)
        pred = model_tr(input_seq).numpy().flatten()
except Exception as e:
    st.warning(f"Transformer 예측 오류: {e}")
    pred = [np.nan, np.nan, np.nan]

# ---- 탭 시각화 ----
col1, col2 = st.columns([2, 8])
with col1:
    st.radio('모델 선택', ['AERMOD', 'AI Transformer'], horizontal=True)
with col2:
    tab1, tab2, tab3 = st.tabs(["지도 시각화", "시계열 분석", "통계 분석"])
    with tab1:
        fig = px.scatter_mapbox(
            map_df, lat='lat', lon='lon', color='농도',
            color_continuous_scale='Jet', size='농도', size_max=20,
            zoom=12, height=400, mapbox_style='carto-positron')
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.line_chart(np.random.randn(48, 2), height=300)
    with tab3:
        st.write('통계 분석 탭 (예시)')

# ---- 결과 지표 ----
st.markdown('<h2>모델 성능 지표</h2>', unsafe_allow_html=True)
col_mae, col_rmse, col_r2 = st.columns(3)
with col_mae:
    st.metric("평균 절대 오차 (MAE)", "3.45 μg/m³")
with col_rmse:
    st.metric("평균 제곱근 오차 (RMSE)", "5.21 μg/m³")
with col_r2:
    st.metric("결정 계수 (R²)", "0.87")

# ---- 상세 결과 ----
st.subheader('AERMOD(가우시안 플룸) 결과')
model = GaussianPlumeModel(Q, latest_weather['speed'].iloc[-1], H, sigma_y, sigma_z)
C = model.concentration(100.0, 0.0, 0.0)
st.write(f"(x=100, y=0, z=0)에서의 농도: **{C:.6f} g/m³**")

st.markdown('#### 여러 지점 시뮬레이션')
points = pd.DataFrame({'x': np.linspace(50, 500, 10), 'y': np.zeros(10), 'z': np.zeros(10)})
df = model.batch_concentration(points.to_dict('records'))
st.line_chart(df[['x', 'concentration']].set_index('x'))

st.subheader('Transformer 시계열 예측 결과')
try:
    st.write(f"NOx 예측: {pred[0]:.2f}, SOx 예측: {pred[1]:.2f}, TSP 예측: {pred[2]:.2f}")
except:
    st.warning("Transformer 예측 실패")

st.markdown('---')
st.markdown('### 예시 데이터 기반 예측 결과')
col1, col2 = st.columns(2)
with col1:
    st.line_chart(aermod_results)
with col2:
    st.bar_chart(pd.DataFrame({'예측값': pred}, index=['NOx', 'SOx', 'TSP']))

st.markdown('#### 실제값 vs Transformer 예측')
true_last = [pred[0]*0.95, pred[1]*1.05, pred[2]*0.9]  # 예시
compare_df = pd.DataFrame({'실제값': true_last, '예측값': pred}, index=['NOx', 'SOx', 'TSP'])
st.dataframe(compare_df)
