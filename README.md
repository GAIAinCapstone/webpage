🧼 데이터 전처리 (prepare_data.py)
본 프로젝트는 굴뚝 자동측정기기(TMS)를 통해 수집된 대기오염물질 데이터를 시계열 예측 모델에 활용하기 위해 전처리를 수행합니다. 전처리는 src/data/prepare_data.py에서 이루어집니다.

📋 주요 처리 단계
1. 데이터 로딩
원격 데이터베이스에서 공장별 오염물질 측정 데이터를 불러옵니다.

2. 날짜 변환
measure_date 컬럼을 datetime64[ns] 형식으로 변환합니다.

정수형 또는 문자열 형식인 경우 "%Y%m%d%H%M%S" 포맷으로 처리됩니다.

변환에 실패한 값(NaT)은 모두 제거합니다.

3. 수치형 변환
'measure', 'stdr'가 포함된 모든 컬럼을 float형으로 변환합니다.

숫자가 아닌 값은 자동으로 NaN 처리됩니다.

4. 결측치 처리
완전 결측 컬럼은 제거합니다.

남은 결측값은 보간(앞→뒤 순)으로 채웁니다.

5. 이상치 처리
수치형 컬럼에 대해 IQR 방식으로 이상치를 처리합니다:

범위: Q1 - 1.5 * IQR ~ Q3 + 1.5 * IQR

벗어난 값은 해당 범위로 클리핑합니다.

6. 시계열 파생 변수 생성
measure_date로부터 다음과 같은 시계열 특성을 생성합니다:

hour: 측정 시각의 시(hour)

day_of_week: 요일 (0=월, 6=일)

month: 측정 월

7. 모델 학습용 데이터 구성
입력값(X): 시간 정보 + 각 오염물질 기준치(*_stdr)

예측값(y): 실제 측정값 (nox, sox, tsp)

8. 결과 저장
전처리 완료된 전체 데이터: data/processed/air_quality_processed.csv

입력 피처: data/processed/features.csv

타겟 변수: data/processed/targets.csv

