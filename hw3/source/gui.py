from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTextEdit, QGraphicsView, QGraphicsScene, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QSlider
)
from PyQt5.QtGui import QPolygonF, QPen, QColor, QPainterPath, QBrush
from PyQt5.QtCore import QPointF, Qt, QTimer
from geometry import parse_track_file, border_to_segments, is_circle_near_segment
from car import Car
import math
import random

class TrackWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("計算型智慧 作業三 MLP+PSO")
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
        self.timer.timeout.connect(self.simulation_step) # 計時器綁定到simulation_step()，每次觸發都會執行這個function
        self.path = QPainterPath()
        self.trajectory_item = None

        # PSO
        self.particle_count_input = None
        self.cognition_rate_input = None
        self.social_rate_input = None
        self.inertia_weight_input = None

    def init_control_panel(self):
        # 匯入座標檔案
        self.import_btn = QPushButton("Import Track File")
        self.import_btn.clicked.connect(self.import_track)
        self.control_layout.addWidget(self.import_btn)

        # PSO 參數調整
        pso_group = QGroupBox("🕊️ PSO Parameters")
        pso_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        pso_layout = QFormLayout()

        # 粒子數
        self.particle_count_input = QLineEdit("30")
        pso_layout.addRow(QLabel("Particle Count:"), self.particle_count_input)

        # 個體學習率
        self.cognition_rate_input = QLineEdit("1.50")
        pso_layout.addRow(QLabel("Cognition Rate:"), self.cognition_rate_input)

        # 社會學習率
        self.social_rate_input = QLineEdit("1.50")
        pso_layout.addRow(QLabel("Social Rate:"), self.social_rate_input)

        # 慣性權重
        self.inertia_weight_input = QLineEdit("0.50")
        pso_layout.addRow(QLabel("Inertia Weight:"), self.inertia_weight_input)

        # 慣性權重
        self.iteration = QLineEdit("100")
        pso_layout.addRow(QLabel("iteration:"), self.iteration)

        pso_group.setLayout(pso_layout)
        self.control_layout.addWidget(pso_group)

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
        self.scene.addPolygon(poly, QPen(QColor("gray"), 1))

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
            pen = QPen(QColor("gray"), 1)
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
        self.timer.start(interval) # 開始計時器

    def stop_simulation(self):
        self.timer.stop()
        self.log_decision("🛑 Simulation manually stopped.")

    def simulation_step(self): # 每次計時器觸發都會執行這個function
        sensor = self.car.get_sensor_distances(border_to_segments(self.border_points))
        action = self.fuzzy_controller.decide_action(sensor)
        self.car.move_forward(action)
        self.update_car_graphics()

        reward, done = self.get_reward()
        if done:
            self.timer.stop()
            self.log_decision("✅ Simulation complete.")

    def log_decision(self, text):
        self.decision_log.append(text)