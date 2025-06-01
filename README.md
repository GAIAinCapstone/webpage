# 대기 오염물질 확산 모델링

이 프로젝트는 가우시안 플룸 모델을 사용하여 대기 오염물질의 확산을 시뮬레이션하는 Python 코드를 포함하고 있습니다.

## 주요 기능

- 가우시안 플룸 모델을 이용한 대기 오염물질 농도 계산
- 다양한 기상 조건과 배출 조건에서의 시뮬레이션 가능

## 사용 방법

```python
from src.models.aermod_simulator import GaussianPlumeModel

# 모델 초기화
model = GaussianPlumeModel(
    Q=100.0,      # 배출량 (g/s)
    u=5.0,        # 풍속 (m/s)
    H=50.0,       # 굴뚝 높이 (m)
    sigma_y=30.0, # y 방향 확산계수 (m)
    sigma_z=15.0  # z 방향 확산계수 (m)
)

# 특정 위치에서의 농도 계산
concentration = model.concentration(x=100.0, y=0.0, z=0.0)
```

## 요구사항

- Python 3.x
- NumPy 