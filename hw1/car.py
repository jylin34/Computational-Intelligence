import math

class Car:
    def __init__(self, x, y, theta=90):
        self.x = x
        self.y = y
        self.theta = theta

    def move_forward(self, step=1):
        rad = math.radians(self.theta)
        self.x += step * math.cos(rad)
        self.y += step * math.sin(rad)

    def rotate(self, delta_angle):
        self.theta += delta_angle
        self.theta %= 360
