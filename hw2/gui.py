from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QGraphicsView, QGraphicsScene, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QSlider
)
from PyQt5.QtGui import QPolygonF, QPen, QColor, QPainterPath, QBrush
from PyQt5.QtCore import QPointF, Qt, QTimer
from geometry import parse_track_file, border_to_segments, is_circle_near_segment
from car import Car
import math
import random

from enum import Enum

class Level(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2

class MembershipFunctions:
    @staticmethod
    def side_small(distance):
        if distance < 10:
            return 1
        elif distance < 12:
            return (12 - distance) / 2
        else:
            return 0

    @staticmethod
    def side_medium(distance):
        if 8 < distance <= 12:
            return (distance - 8) / 4
        elif 12 < distance <= 16:
            return (16 - distance) / 4
        else:
            return 0

    @staticmethod
    def side_large(distance):
        if 13 < distance <= 20:
            return (distance - 13) / 7
        elif distance > 20:
            return 1
        else:
            return 0

    @staticmethod
    def front_small(distance):
        if distance < 10:
            return 1
        elif distance < 15:
            return (15 - distance) / 5
        else:
            return 0

    @staticmethod
    def front_medium(distance):
        if 19 < distance <= 21:
            return (distance - 19) / 2
        elif 21 < distance <= 23:
            return (23 - distance) / 2
        else:
            return 0

    @staticmethod
    def front_large(distance):
        if distance > 30:
            return 1
        else:
            return 0

class Fuzzifier:
    @staticmethod
    def to_level(s, m, l):
        return Level([s, m, l].index(max([s, m, l])))

    @staticmethod
    def l_point(distance):
        s = MembershipFunctions.side_small(distance)
        m = MembershipFunctions.side_medium(distance)
        l = MembershipFunctions.side_large(distance)
        return Fuzzifier.to_level(s, m, l)

    @staticmethod
    def r_point(distance):
        s = MembershipFunctions.side_small(distance)
        m = MembershipFunctions.side_medium(distance)
        l = MembershipFunctions.side_large(distance)
        return Fuzzifier.to_level(s, m, l)

    @staticmethod
    def c_point(distance):
        s = MembershipFunctions.front_small(distance)
        m = MembershipFunctions.front_medium(distance)
        l = MembershipFunctions.front_large(distance)
        return Fuzzifier.to_level(s, m, l)

class Rules:
    @staticmethod
    def apply(l_point, c_point, r_point):
        if r_point == Level.SMALL:
            return -40
        if l_point == Level.SMALL:
            return 40
        if r_point == Level.MEDIUM and c_point == Level.SMALL:
            return -20
        if l_point == Level.MEDIUM and c_point == Level.SMALL:
            return 20
        return 0

class FuzzyController:
    def __init__(self):
        self.fuzzifier = Fuzzifier()

    def decide_action(self, sensor_data):
        right, front, left = sensor_data
        l_point = self.fuzzifier.l_point(left)
        c_point = self.fuzzifier.c_point(front)
        r_point = self.fuzzifier.r_point(right)
        return Rules.apply(l_point, c_point, r_point)

class TrackWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("計算型智慧 作業二 Fuzzy System")
        self.setGeometry(100, 100, 900, 550)

        # 建立主畫面 layout（水平切左右）
        main_layout = QHBoxLayout(self)

        # 左側：控制區
        self.control_layout = QVBoxLayout()
        self.param_layout = QFormLayout()
        self.init_control_panel()
        main_layout.addLayout(self.control_layout, 2)  # 左邊佔 1 份寬度
        self.control_layout.addLayout(self.param_layout)

        # 右側：畫布區
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.scale(2, 2)
        main_layout.addWidget(self.view, 3)  # 右邊佔 3 份寬度

        self.car = None
        self.car_item = None
        self.car_dir_line = None
        self.border_points = []
        self.SCALE = 4
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)
        self.path = QPainterPath()
        self.trajectory_item = None

        self.fuzzy_controller = FuzzyController()  # 初始化模糊控制器

    def init_control_panel(self):
        # 匯入座標檔案
        self.import_btn = QPushButton("Import Track File")
        self.import_btn.clicked.connect(self.import_track)
        self.control_layout.addWidget(self.import_btn)

        # 執行速度調整 + 顯示文字
        self.speed_label = QLabel("Simulation Speed: 1 ms")
        self.speed_label.setStyleSheet("font-weight: bold;")
        self.control_layout.addWidget(self.speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(1)  # 預設間隔 1ms
        self.speed_slider.setTickInterval(50)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)

        self.speed_slider.valueChanged.connect(
            lambda value: self.speed_label.setText(f"Simulation Speed: {value} ms")
        )

        self.control_layout.addWidget(self.speed_slider)

        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self.start_simulation)
        self.control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Simulation")
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.control_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("Reset Car")
        self.reset_btn.clicked.connect(self.reset_car)
        self.control_layout.addWidget(self.reset_btn)

        # 決策紀錄
        self.decision_log = QTextEdit()
        self.decision_log.setReadOnly(True)
        self.control_layout.addWidget(QLabel("Decision Log"))
        self.control_layout.addWidget(self.decision_log)

        car_group = QGroupBox("🚗 Car Information")
        car_group.setStyleSheet("QGroupBox { font-weight: bold; }")

        self.car_info_grid = QGridLayout()
        self.car_x_label = QLabel("X: 0.00")
        self.car_y_label = QLabel("Y: 0.00")
        self.car_theta_label = QLabel("θ: 0.00°")

        for lbl in [self.car_x_label, self.car_y_label, self.car_theta_label]:
            lbl.setFixedWidth(80)
            lbl.setStyleSheet("font-family: monospace; padding: 4px; border: 1px solid #ccc;")
            lbl.setAlignment(Qt.AlignCenter)

        self.car_info_grid.addWidget(self.car_x_label, 0, 0)
        self.car_info_grid.addWidget(self.car_y_label, 0, 1)
        self.car_info_grid.addWidget(self.car_theta_label, 0, 2)
        car_group.setLayout(self.car_info_grid)

        self.control_layout.addWidget(car_group)

        sensor_group = QGroupBox("📡 Sensor Information")
        sensor_group.setStyleSheet("QGroupBox { font-weight: bold; }")

        self.sensor_info_grid = QGridLayout()
        self.sensor_left_label = QLabel("Left: 0.00")
        self.sensor_front_label = QLabel("Front: 0.00")
        self.sensor_right_label = QLabel("Right: 0.00")

        for lbl in [self.sensor_left_label, self.sensor_front_label, self.sensor_right_label]:
            lbl.setFixedWidth(80)
            lbl.setStyleSheet("font-family: monospace; padding: 4px; border: 1px solid #ccc;")
            lbl.setAlignment(Qt.AlignCenter)

        self.sensor_info_grid.addWidget(self.sensor_left_label, 0, 0)
        self.sensor_info_grid.addWidget(self.sensor_front_label, 0, 1)
        self.sensor_info_grid.addWidget(self.sensor_right_label, 0, 2)
        sensor_group.setLayout(self.sensor_info_grid)

        self.control_layout.addWidget(sensor_group)

    def import_track(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Track File", "", "Text Files (*.txt)")
        if not path:
            return

        start, start_tl, start_br, goal_tl, goal_br, border = parse_track_file(path)
        self.start = start
        self.goal_tl = goal_tl
        self.goal_br = goal_br
        self.draw_track(start, start_tl, start_br, goal_tl, goal_br, border)

    def draw_track(self, start, start_tl, start_br, goal_tl, goal_br, border_points):
        self.border_points = border_points
        self.scene.clear()

        # 畫邊界
        poly = QPolygonF([QPointF(x * self.SCALE, -y * self.SCALE) for x, y in border_points])
        self.scene.addPolygon(poly, QPen(QColor("white"), 1))

        # 畫起點
        self.scene.addEllipse(start[0] * self.SCALE - 3, -start[1] * self.SCALE - 3, 6, 6, brush=QColor("red"))

        # 畫終點
        x1, y1 = goal_tl
        x2, y2 = goal_br
        goal_poly = QPolygonF([
            QPointF(x1 * self.SCALE, -y1 * self.SCALE),
            QPointF(x2 * self.SCALE, -y1 * self.SCALE),
            QPointF(x2 * self.SCALE, -y2 * self.SCALE),
            QPointF(x1 * self.SCALE, -y2 * self.SCALE),
        ])
        self.scene.addPolygon(goal_poly, QPen(QColor("green"), 1, style=3))

        # 初始化車輛
        self.car = Car(start[0] * self.SCALE, start[1] * self.SCALE, theta=start[2])
        self.update_car_graphics()

    def reset_car(self):
        if self.car_item:
            self.scene.removeItem(self.car_item)
        if self.car_dir_line:
            self.scene.removeItem(self.car_dir_line)

        random_x = random.uniform(-3, 3)
        theta = 90
        self.car = Car(random_x, self.start[1], theta=theta)

        if self.trajectory_item:
            self.scene.removeItem(self.trajectory_item)
            self.trajectory_item = None
        self.path = QPainterPath()
        x = self.car.x * self.SCALE
        y = -self.car.y * self.SCALE
        self.path.moveTo(x, y)

        self.update_car_graphics()

    def update_car_graphics(self):
        if self.car_item:
            self.scene.removeItem(self.car_item)
        if self.car_dir_line:
            self.scene.removeItem(self.car_dir_line)

        # 更新車輛資訊
        self.car_x_label.setText(f"X: {self.car.x:.2f}")
        self.car_y_label.setText(f"Y: {self.car.y:.2f}")
        self.car_theta_label.setText(f"θ: {self.car.theta:.1f}°")

        # 更新感測器資訊
        sensor = self.car.get_sensor_distances(border_to_segments(self.border_points))
        self.sensor_left_label.setText(f"Left: {sensor[2]:.2f}")
        self.sensor_front_label.setText(f"Front: {sensor[1]:.2f}")
        self.sensor_right_label.setText(f"Right: {sensor[0]:.2f}")

        # 更新軌跡線
        x = self.car.x * self.SCALE
        y = -self.car.y * self.SCALE
        self.path.lineTo(x, y)

        if self.trajectory_item:
            self.trajectory_item.setPath(self.path)
        else:
            pen = QPen(QColor("white"), 1)
            self.trajectory_item = self.scene.addPath(self.path, pen)

        pen = QPen(QColor(0, 0, 255, 70))
        brush = QBrush(QColor(0, 0, 255, 30))

        self.car_item = self.scene.addEllipse(
            self.car.x * self.SCALE - 3 * self.SCALE,
            -self.car.y * self.SCALE - 3 * self.SCALE,
            6 * self.SCALE,
            6 * self.SCALE,
            pen,
            brush
        )

        rad = math.radians(self.car.theta)
        x2 = self.car.x + math.cos(rad) * 1.0
        y2 = self.car.y + math.sin(rad) * 1.0
        pen = QPen(QColor(0, 255, 255, 70), 1)
        self.car_dir_line = self.scene.addLine(
            self.car.x * self.SCALE,
            -self.car.y * self.SCALE,
            x2 * self.SCALE + 3 * self.SCALE * math.cos(rad),
            -y2 * self.SCALE - 3 * self.SCALE * math.sin(rad),
            pen
        )

    def start_simulation(self):
        interval = self.speed_slider.value()
        self.timer.start(interval)

    def stop_simulation(self):
        self.timer.stop()
        self.log_decision("🛑 Simulation manually stopped.")

    def simulation_step(self):
        sensor = self.car.get_sensor_distances(border_to_segments(self.border_points))
        action = self.fuzzy_controller.decide_action(sensor)
        self.car.move_forward(action)
        self.update_car_graphics()

        reward, done = self.get_reward()
        if done:
            self.timer.stop()
            self.log_decision("✅ Simulation complete.")

    def get_reward(self):
        x, y = self.car.x, self.car.y
        radius = 3

        gx1, gy1 = self.goal_tl
        gx2, gy2 = self.goal_br
        if gx1 <= x <= gx2 and gy2 <= y <= gy1:
            return 1000, True

        for x1, y1, x2, y2 in border_to_segments(self.border_points):
            if is_circle_near_segment(x, y, radius, x1, y1, x2, y2):
                return -100, True

        return 1, False

    def log_decision(self, text):
        self.decision_log.append(text)


class FuzzyController:
    def __init__(self):
        self.fuzzifier = Fuzzifier()

    def decide_action(self, sensor_data):
        right, front, left = sensor_data
        l_point = self.fuzzifier.l_point(left)
        c_point = self.fuzzifier.c_point(front)
        r_point = self.fuzzifier.r_point(right)
        return Rules.apply(l_point, c_point, r_point)