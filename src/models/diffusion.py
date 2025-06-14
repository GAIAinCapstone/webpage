class DiffusionCoefficient:
    @staticmethod
    def _cal(x, a, b, c):
        return a * (x**c) + b

    def calculation_y(self, x, stability):  # y 확산계수 반환 | 매개변수: 거리, 대기안정도
        y_params = [
            [0.22, 0.16, 0.11, 0.08, 0.06, 0.04],  # a
            [0.0001 for _ in range(6)],            # b
            [-0.5 for _ in range(6)]               # c
        ]
        idx = ord(stability.upper()) - ord('A')
        return self._cal(x, y_params[0][idx], y_params[1][idx], y_params[2][idx])

    def calculation_z(self, x, stability):  # z 확산계수 반환
        z_params = [
            [0.2, 0.12, 0.08, 0.06, 0.03, 0.016],     # a
            [0, 0, 0.0002, 0.0015, 0.0003, 0.0003],   # b
            [1, 1, -0.5, -0.5, -1, -1]               # c
        ]
        idx = ord(stability.upper()) - ord('A')
        return self._cal(x, z_params[0][idx], z_params[1][idx], z_params[2][idx])

    @staticmethod
    def classify_insolation(insolation):    #일사량 계산 | 매개변수 단위 W/m^2
        if insolation >= 700:
            return 'strong'
        elif insolation >= 350:
            return 'moderate'
        else:
            return 'slight'

    @staticmethod
    def convert_cloudiness_to_okta(cloud_tenth):    #전운량 10분율 -> 8옥타
        return (8 * cloud_tenth) / 10

    @classmethod
    def classify_cloudiness(cls, cloud_tenth):      #전운량 계산 | 매개변수: 10분율 전운량
        okta = cls.convert_cloudiness_to_okta(cloud_tenth)
        return 'clear' if okta <= 4 else 'cloudy'

    @staticmethod
    def get_stability(wind_speed, condition, is_daytime):  #대기안정도 계산 | 매개변수: 풍속 10m 기분 m/s, 일사량(낮일 때) or 전운량(밤일 때), 낮(true) or 밤(false)
        if is_daytime:
            if condition == 'strong':
                if wind_speed < 2: return 'A'
                elif wind_speed < 3: return 'A'
                elif wind_speed < 5: return 'B'
                elif wind_speed < 6: return 'C'
                else: return 'C'
            elif condition == 'moderate':
                if wind_speed < 2: return 'B'
                elif wind_speed < 3: return 'B'
                elif wind_speed < 5: return 'B'
                elif wind_speed < 6: return 'C'
                else: return 'D'
            elif condition == 'slight':
                if wind_speed < 2: return 'C'
                elif wind_speed < 3: return 'C'
                elif wind_speed < 5: return 'C'
                elif wind_speed < 6: return 'D'
                else: return 'D'
            else:
                raise ValueError("condition은 'strong', 'moderate', 'slight' 중 하나여야 합니다.")
        else:
            if condition == 'clear':
                if wind_speed < 2: return 'F'
                elif wind_speed < 3: return 'E'
                elif wind_speed < 5: return 'D'
                else: return 'D'
            elif condition == 'cloudy':
                if wind_speed < 2: return 'E'
                elif wind_speed < 3: return 'D'
                elif wind_speed < 5: return 'D'
                else: return 'D'
            else:
                raise ValueError("condition은 'clear', 'cloudy' 중 하나여야 합니다.")
'''
y확산계수 a,b,c | z확산계수 a, b, c
A-F는 대기안정도

Open-Country Conditions
A: 0.22 0.0001 -1/2      0.20 0 1
B: 0.16 0.0001 -1/2      0.12 0 1
C: 0.11 0.0001 -1/2      0.08 0.0002 -1/2
D: 0.08 0.0001 -1/2      0.06 0.0015 -1/2
E: 0.06 0.0001 -1/2      0.03 0.0003 -1
F: 0.04 0.0001 -1/2      0.016 0.0003 -1

Urban Conditions
A-B: 0.32 0.0004 -1/2       0.24 0.001 1/2
C:   0.22 0.0004 -1/2       0.2 0 1
D:   0.16 0.0004 -1/2       0.14 0.0003 -1/2
E-F: 0.11 0.0004 -1/2       0.08 0.00015 -1/2
'''