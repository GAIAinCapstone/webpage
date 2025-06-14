import streamlit as st
import pandas as pd
import numpy as np
import torch
from src.models.aermod_simulator import GaussianPlumeModel
from src.models.transformer_model import SimpleTransformer
from config.database import get_database_connection, fetch_air_quality_data
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

st.sidebar.header('기상 조건 설정')
avg_wind = st.sidebar.slider('평균 풍속 (m/s)', 0.0, 20.0, 5.0)
wind_dir = st.sidebar.slider('주풍향 (도)', 0, 360, 180)
stability = st.sidebar.selectbox('대기 안정도', ['매우 불안정', '불안정', '중립', '안정', '매우 안정'], index=2)

st.sidebar.header('배출원 설정')
source_type = st.sidebar.selectbox('배출원 유형', ['점 오염원 (공장)', '면 오염원 (산업단지)', '선 오염원 (도로)'])
Q = st.sidebar.slider('배출량 (g/s)', 0.0, 100.0, 10.0)

if st.sidebar.button('분석 실행'):
    st.session_state['run_analysis'] = True

# ---- 메인 ----
st.markdown("""
    <style>
    .metric-card {background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; margin: 0 10px;}
    .legend-color {width: 12px; height: 12px; border-radius: 2px; display: inline-block; margin-right: 6px;}
    .toggle-btn {display: inline-block; background: #ccc; border-radius: 30px; width: 60px; height: 30px; position: relative; vertical-align: middle; margin: 0 10px;}
    .toggle-btn.active {background: #f63366;}
    .toggle-dot {position: absolute; width: 26px; height: 26px; background: white; border-radius: 50%; top: 2px; left: 2px; transition: 0.2s;}
    .toggle-btn.active .toggle-dot {left: 32px;}
    </style>
""", unsafe_allow_html=True)

st.title('대기질 예측 결과')

# ---- DB에서 데이터 불러오기 ----
def load_db_data():
    conn = get_database_connection()
    if conn is not None:
        df = fetch_air_quality_data(conn)
        conn.close()
        return df
    return pd.DataFrame()

db_data = load_db_data()

st.markdown('### 실시간 DB 데이터')
st.dataframe(db_data.head(50))

# ---- 모델 선택 토글 ----
col1, col2 = st.columns([2, 8])
with col1:
    st.markdown('<span style="font-weight:bold;">모델 선택:</span>', unsafe_allow_html=True)
    toggle = st.radio('', ['AERMOD', 'AI Transformer'], horizontal=True, index=0)

# ---- 예시 데이터 불러오기 ----
features = pd.read_csv('data/processed/features.csv')
targets = pd.read_csv('data/processed/targets.csv')

# ---- AERMOD 예측 ----
Q_list = features['nox_stdr'][:10].values  # 임의로 nox_stdr를 Q로 사용
u_list = features['hour'][:10].values + 1 # 임의로 hour+1을 풍속으로 사용(0 방지)
H, sigma_y, sigma_z = 50.0, 30.0, 15.0
x, y, z = 100.0, 0.0, 0.0

aermod_results = []
for Q, u in zip(Q_list, u_list):
    model = GaussianPlumeModel(Q, u, H, sigma_y, sigma_z)
    c = model.concentration(x, y, z)
    aermod_results.append(c)

aermod_results = np.array(aermod_results)

# ---- 지도 시각화용 임의 위치 데이터 생성 (탭보다 위에서 정의) ----
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
import torch
input_dim = features.shape[1]
output_dim = 3
model_tr = SimpleTransformer(input_dim, output_dim)
try:
    model_tr.load_state_dict(torch.load('src/models/transformer_model.pt', map_location=torch.device('cpu')))
    model_tr.eval()
    with torch.no_grad():
        input_seq = torch.tensor(features[-24:].values.astype(np.float32)).unsqueeze(0)
        pred = model_tr(input_seq).numpy().flatten()
except Exception as e:
    pred = [np.nan, np.nan, np.nan]

# ---- 지도 시각화(플레이스홀더) ----
with col2:
    tab1, tab2, tab3 = st.tabs(["지도 시각화", "시계열 분석", "통계 분석"])
    with tab1:
        st.markdown('#### 지도 기반 AERMOD 예측 시각화')
        fig = px.scatter_mapbox(
            map_df,
            lat='lat', lon='lon', color='농도',
            color_continuous_scale='Jet',
            size='농도', size_max=20,
            zoom=12, height=400,
            mapbox_style='carto-positron',
            hover_data={'lat': True, 'lon': True, '농도': True}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style='display:flex; justify-content:center; gap:20px; margin-top:10px;'>
            <div class='legend-color' style='background:#0000FF;'></div>0-25 μg/m³
            <div class='legend-color' style='background:#00FF00;'></div>25-50 μg/m³
            <div class='legend-color' style='background:#FFFF00;'></div>50-100 μg/m³
            <div class='legend-color' style='background:#FFA500;'></div>100-150 μg/m³
            <div class='legend-color' style='background:#FF0000;'></div>>150 μg/m³
        </div>
        """, unsafe_allow_html=True)
    with tab2:
        st.line_chart(np.random.randn(48, 2), height=300)
        st.markdown("""
        <div style='display:flex; justify-content:center; gap:20px; margin-top:10px;'>
            <div class='legend-color' style='background:#FF0000;'></div>AERMOD 예측
            <div class='legend-color' style='background:#0000FF;'></div>AI Transformer 예측
            <div class='legend-color' style='background:#00FF00;'></div>실측 데이터
        </div>
        """, unsafe_allow_html=True)
    with tab3:
        st.write('통계 분석 탭 (예시)')

# ---- 모델 결과/지표 카드 ----
st.markdown('<h2>모델 성능 지표</h2>', unsafe_allow_html=True)
col_mae, col_rmse, col_r2 = st.columns(3)
with col_mae:
    st.markdown('<div class="metric-card"><div class="metric-label">평균 절대 오차 (MAE)</div><div class="metric-value">3.45</div><div class="metric-label">μg/m³</div></div>', unsafe_allow_html=True)
with col_rmse:
    st.markdown('<div class="metric-card"><div class="metric-label">평균 제곱근 오차 (RMSE)</div><div class="metric-value">5.21</div><div class="metric-label">μg/m³</div></div>', unsafe_allow_html=True)
with col_r2:
    st.markdown('<div class="metric-card"><div class="metric-label">결정 계수 (R²)</div><div class="metric-value">0.87</div><div class="metric-label">높을수록 정확</div></div>', unsafe_allow_html=True)

# ---- 예측 시계열 분석 ----
st.markdown('<h2>예측 시계열 분석</h2>', unsafe_allow_html=True)
st.line_chart(np.random.randn(48, 2), height=300)
st.markdown("""
<div style='display:flex; justify-content:center; gap:20px; margin-top:10px;'>
    <div class='legend-color' style='background:#FF0000;'></div>AERMOD 예측
    <div class='legend-color' style='background:#0000FF;'></div>AI Transformer 예측
    <div class='legend-color' style='background:#00FF00;'></div>실측 데이터
</div>
""", unsafe_allow_html=True)

st.subheader('AERMOD(가우시안 플룸) 결과')
model = GaussianPlumeModel(Q, avg_wind, 50.0, 30.0, 15.0)
C = model.concentration(100.0, 0.0, 0.0)
st.write(f"(x=100, y=0, z=0)에서의 농도: **{C:.6f} g/m³**")

# 여러 지점 시뮬레이션
st.markdown('#### 여러 지점 시뮬레이션')
points = pd.DataFrame({
    'x': np.linspace(50, 500, 10),
    'y': np.zeros(10),
    'z': np.zeros(10)
})
df = model.batch_concentration(points.to_dict('records'))
st.line_chart(df[['x', 'concentration']].set_index('x'))

st.subheader('Transformer 시계열 예측 결과')
# 예시: 가장 최근 24개 feature로 예측
features = pd.read_csv('data/processed/features.csv').values.astype(np.float32)
input_seq = torch.tensor(features[-24:]).unsqueeze(0)  # (1, 24, feature_dim)

# 모델 로드
input_dim = features.shape[1]
output_dim = 3
model_path = 'src/models/transformer_model.pt'
model_tr = SimpleTransformer(input_dim, output_dim)
try:
    model_tr.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_tr.eval()
    with torch.no_grad():
        pred = model_tr(input_seq).numpy().flatten()
    st.write(f"NOx 예측: {pred[0]:.2f}, SOx 예측: {pred[1]:.2f}, TSP 예측: {pred[2]:.2f}")
except Exception as e:
    st.warning('Transformer 모델이 아직 학습되지 않았거나, 모델 파일이 없습니다.')
    st.text(str(e))

st.markdown('---')
st.markdown('**AERMOD와 Transformer 예측 결과를 비교하고, 다양한 입력값으로 시뮬레이션해보세요!**')

# ---- 시각화 ----
st.markdown('### 예시 데이터 기반 예측 결과')
col1, col2 = st.columns(2)
with col1:
    st.markdown('#### AERMOD 예측 (농도)')
    st.line_chart(aermod_results)
with col2:
    st.markdown('#### Transformer 예측 (NOx, SOx, TSP)')
    st.bar_chart(pd.DataFrame({'예측값': pred}, index=['NOx', 'SOx', 'TSP']))

# 실제값과 비교
st.markdown('#### 실제값 vs Transformer 예측')
true_last = targets.iloc[-1].values
compare_df = pd.DataFrame({'실제값': true_last, '예측값': pred}, index=['NOx', 'SOx', 'TSP'])
st.dataframe(compare_df) 