import math
from geometry import cast_ray

class Car:
    def __init__(self, x, y, theta=90): # 初始角度為90度
        self.x = x
        self.y = y
        self.theta = theta

    def move_forward(self, angle):
        rad = math.radians(self.theta + angle)
        self.x += math.cos(rad) + math.sin(math.radians(self.theta)) * math.sin(math.radians(angle))
        self.y += math.sin(rad) - math.sin(math.radians(self.theta)) * math.cos(math.radians(angle))

    def rotate(self, delta_angle):
        self.theta = self.normalize_angle(self.theta + delta_angle)

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
