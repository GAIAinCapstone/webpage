import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. 전처리된 파일 불러오기
    df = pd.read_csv('data/processed/air_quality_processed.csv')
    
    # 👉 datetime 복원
    df['measure_date'] = pd.to_datetime(df['measure_date'], errors='coerce')

    print("✅ 파일 로드 성공")
    print("\n📄 데이터 타입 확인:")
    print(df.dtypes)

    print("\n🕳️ 결측치 확인:")
    print(df.isna().sum())

    print("\n📊 수치형 요약 통계:")
    print(df.describe())

    print("\n📆 시간 관련 컬럼 확인:")
    for col in ['hour', 'day_of_week', 'month']:
        if col in df.columns:
            print(f"✅ {col} 컬럼 존재")
        else:
            print(f"❌ {col} 컬럼 없음")

    # 2. 분포 시각화 (주요 오염물질)
    plot_columns = ['nox_measure', 'sox_measure', 'tsp_measure']
    for col in plot_columns:
        if col in df.columns:
            df[col].hist(bins=50)
            plt.title(f'{col} 분포 (전처리 후)')
            plt.xlabel(col)
            plt.ylabel('빈도')
            plt.grid(False)
            plt.show()
        else:
            print(f"⚠️ {col} 컬럼 없음 - 그래프 생략")

if __name__ == "__main__":
    main()
