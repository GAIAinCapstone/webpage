import numpy as np
import pandas as pd

class GaussianPlumeModel:
    """
    가우시안 플룸(확산) 모델을 이용한 대기 오염물질 농도 계산 클래스
    """
    def __init__(self, Q, u, H, sigma_y, sigma_z):
        """
        Args:
            Q (float): 배출량 (g/s)
            u (float): 풍속 (m/s)
            H (float): 굴뚝 높이 (m)
            sigma_y (float): y 방향 확산계수 (m)
            sigma_z (float): z 방향 확산계수 (m)
        """
        self.Q = Q
        self.u = u
        self.H = H
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z

    def concentration(self, x, y, z=0):
        """
        (x, y, z) 위치에서의 오염물질 농도를 계산합니다.
        Args:
            x (float): 풍향을 따라 다운윈드 거리 (m)
            y (float): 중심선에서의 횡방향 거리 (m)
            z (float): 지상에서의 높이 (m, 기본값=0)
        Returns:
            float: 농도 (g/m^3)
        """
        Q = self.Q
        u = self.u
        H = self.H
        sigma_y = self.sigma_y
        sigma_z = self.sigma_z

        part1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
        part2 = np.exp(-y**2 / (2 * sigma_y**2))
        part3 = np.exp(-(z - H)**2 / (2 * sigma_z**2))
        part4 = np.exp(-(z + H)**2 / (2 * sigma_z**2))
        C = part1 * part2 * (part3 + part4)
        return C

    def batch_concentration(self, points):
        """
        여러 지점에 대해 농도를 계산하여 DataFrame으로 반환합니다.
        Args:
            points (list of dict): [{'x':..., 'y':..., 'z':...}, ...]
        Returns:
            DataFrame: 각 지점별 농도 결과
        """
        results = []
        for pt in points:
            x, y, z = pt['x'], pt['y'], pt.get('z', 0)
            c = self.concentration(x, y, z)
            results.append({'x': x, 'y': y, 'z': z, 'concentration': c})
        return pd.DataFrame(results)

# 예시 사용법
if __name__ == "__main__":
    # 예시 파라미터
    Q = 100.0      # g/s
    u = 5.0        # m/s
    H = 50.0       # m
    sigma_y = 30.0 # m
    sigma_z = 15.0 # m

    model = GaussianPlumeModel(Q, u, H, sigma_y, sigma_z)
    # 여러 지점 예시
    points = [
        {'x': 100, 'y': 0, 'z': 0},
        {'x': 200, 'y': 0, 'z': 0},
        {'x': 300, 'y': 0, 'z': 0},
        {'x': 100, 'y': 50, 'z': 0},
        {'x': 100, 'y': -50, 'z': 0},
    ]
    df = model.batch_concentration(points)
    print(df) 