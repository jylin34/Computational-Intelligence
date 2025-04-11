import math
from geometry import cast_ray

class Car:
    def __init__(self, x, y, theta=90): # 初始角度為90度
        self.x = x
        self.y = y
        self.theta = theta

    def move_forward(self, step=1):
        rad = math.radians(self.theta)
        self.x += step * math.cos(rad)
        self.y += step * math.sin(rad)

    def rotate(self, delta_angle):
        self.theta = normalize_angle(self.theta + delta_angle)

    def normalize_angle(theta): # phi: -90 ~ 270
        while theta < -90:
            theta += 360
        while theta >= 270:
            theta -= 360
        return theta

    def get_sensor_distance(self, border_segments):
        angles = [self.theta - 45, self.theta, self.theta + 45]
        return [
            cast_ray(self.x, self.y, angle, border_segments)
            for angle in angles
        ]
