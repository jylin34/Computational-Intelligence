import math
from geometry import cast_ray

class Car:
    def __init__(self, x, y, theta=90): # 初始角度為90度
        self.x = x
        self.y = y
        self.theta = theta

    def move_forward(self, theta, step=1, wheel_base=6):
        rad_phi = math.radians(self.theta)
        rad_theta = math.radians(theta)

        # 更新 x
        self.x += math.cos(rad_phi + rad_theta) + math.sin(rad_theta) * math.sin(rad_phi)

        # 更新 y
        self.y += math.sin(rad_phi + rad_theta) - math.sin(rad_theta) * math.cos(rad_phi)

        # 更新 heading 角度（phi）
        delta_phi = math.degrees(math.asin(2 * math.sin(rad_theta) / wheel_base))
        self.theta -= delta_phi
        self.theta = self.normalize_angle(self.theta)

    def rotate(self, delta_angle):
        # self.theta = self.normalize_angle(self.theta + delta_angle)
        self.theta -= math.asin(2 * math.sin(-delta_angle) / 6)

    def normalize_angle(self, theta): # phi: -90 ~ 270
        while theta < -90:
            theta += 360
        while theta >= 270:
            theta -= 360
        return theta

    def get_sensor_distances(self, border_segments):
        angles = [self.theta - 45, self.theta, self.theta + 45]
        return [
            cast_ray(self.x, self.y, angle, border_segments)
            for angle in angles
        ]
