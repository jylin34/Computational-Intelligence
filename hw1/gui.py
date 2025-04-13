from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTextEdit, QGraphicsView, QGraphicsScene, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QSlider
)
from PyQt5.QtGui import QPolygonF, QPen, QColor, QPainterPath, QBrush
from PyQt5.QtCore import QPointF, Qt, QTimer
from geometry import parse_track_file, border_to_segments, is_circle_near_segment
from car import Car
import math
from agent import Agent

class TrackWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("計算型智慧 作業一 Q-learning")
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
        
        self.agent = None
        self.car = None
        self.car_item = None
        self.car_dir_line = None
        self.border_points = []
        self.SCALE = 4
        self.STEP = 1
        self.current_episode = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.train_step)
        self.path = QPainterPath()           # 記錄所有點
        self.trajectory_item = None          # 對應的 QG
        self.is_testing = False
        self.step_reward = 1
        # self.agent = Agent(
        #    lr=float(self.lr_input.text()),
        #    discount_factor=float(self.discounted_factor.text()),
        #    epsilon=float(self.eps_input.text()),
        #    epsilon_decay=float(self.epsd_input.text())
        #)

    def init_control_panel(self):
        # 匯入座標檔案
        self.import_btn = QPushButton("Import Track File")
        self.import_btn.clicked.connect(self.import_track)
        self.control_layout.addWidget(self.import_btn)

        # Learning Rate
        self.lr_input = QLineEdit("0.31")
        self.param_layout.addRow(QLabel("Learning Rate"), self.lr_input)

        # Epsilon
        self.eps_input = QLineEdit("1")
        self.param_layout.addRow(QLabel("Epsilon"), self.eps_input)

        # Epsilon Decay
        self.epsd_input = QLineEdit("0.995")
        self.param_layout.addRow(QLabel("Epsilon Decay"), self.epsd_input)

        # Discount Factor
        self.discounted_factor = QLineEdit("0.95")
        self.param_layout.addRow(QLabel("Discount Factor"), self.discounted_factor)

        # Step
        # self.step = QLineEdit("500")
        # self.param_layout.addRow(QLabel("Step"), self.step)

        # Episode
        self.episode_label = QLineEdit("3000")
        self.param_layout.addRow(QLabel("Episode"), self.episode_label)

        self.control_layout.addLayout(self.param_layout)

        # 執行速度調整 + 顯示文字
        self.speed_label = QLabel("Training Speed: 1 ms")
        self.speed_label.setStyleSheet("font-weight: bold;")
        self.control_layout.addWidget(self.speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(1)  # 預設間隔 100ms
        self.speed_slider.setTickInterval(50)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)

        self.speed_slider.valueChanged.connect(
            lambda value: self.speed_label.setText(f"Training Speed: {value} ms")
        )

        self.control_layout.addWidget(self.speed_slider)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
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
        # 車子的部分
        # self.car = None
        # self.car_item = None
        # self.car_dir_line = None

    def import_track(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Track File", "", "Text Files (*.txt)")
        if not path:
            return

        # 呼叫 geometry.py 中的解析函式
        start, start_tl, start_br, goal_tl, goal_br, border = parse_track_file(path)
        self.start = start
        self.goal_tl = goal_tl
        self.goal_br = goal_br
        self.draw_track(start, start_tl, start_br, goal_tl, goal_br, border)

    def draw_track(self, start, start_tl, start_br, goal_tl, goal_br, border_points):
        # axis_pen = QPen(QColor("gray"))
        # axis_pen.setStyle(Qt.DashLine)  # 虛線更不干擾畫面

        # X 軸：從左到右
        # self.scene.addLine(-1000, 0, 1000, 0, axis_pen)

        # Y 軸：從上到下
        # self.scene.addLine(0, -1000, 0, 1000, axis_pen)

        self.border_points = border_points

        self.scene.clear()

        # 畫邊界（黑線）
        poly = QPolygonF([QPointF(x * self.SCALE, -y * self.SCALE) for x, y in border_points])
        self.scene.addPolygon(poly, QPen(QColor("white"), 1))

        # 畫起點（紅點）
        self.scene.addEllipse(start[0] * self.SCALE - 3, -start[1] * self.SCALE - 3, 6, 6, brush=QColor("red"))

        # 起點線
        x1, y1 = start_tl
        x2, y2 = start_br
        self.scene.addLine(x1 * self.SCALE, -y1 * self.SCALE, x2 * self.SCALE, -y2 * self.SCALE, QPen(QColor("gray"), 1))

        # 畫終點（綠框）
        x1, y1 = goal_tl
        x2, y2 = goal_br
        goal_poly = QPolygonF([
            QPointF(x1 * self.SCALE, -y1 * self.SCALE),
            QPointF(x2 * self.SCALE, -y1 * self.SCALE),
            QPointF(x2 * self.SCALE, -y2 * self.SCALE),
            QPointF(x1 * self.SCALE, -y2 * self.SCALE),
        ])
        self.scene.addPolygon(goal_poly, QPen(QColor("green"), 1, style=3))

        # 車子圓形
        self.car = Car(start[0] * self.SCALE, start[1] * self.SCALE, theta=start[2])
        self.car_item = self.scene.addEllipse(self.car.x * self.SCALE - 3 * self.SCALE, -self.car.y * self.SCALE - 3 * self.SCALE, 6 * self.SCALE, 6 * self.SCALE, QPen(QColor("blue")))
        rad = math.radians(self.car.theta) # 90 -> pi/2
        x2 = self.car.x + math.cos(rad) * 1.0
        y2 = self.car.y + math.sin(rad) * 1.0
        
        # 車子指向
        self.car_dir_line = self.scene.addLine(self.car.x * self.SCALE, -self.car.y * self.SCALE, x2 * self.SCALE, -y2 * self.SCALE - 3 * self.SCALE, QPen(QColor("cyan"), 1))

    def reset_car(self):
        if self.car_item:
            self.scene.removeItem(self.car_item)
        if self.car_dir_line:
            self.scene.removeItem(self.car_dir_line)
        # theta = random.choice([90, 0, -45])
        theta = 90
        self.car = Car(self.start[0] * self.SCALE, self.start[1] * self.SCALE, theta=theta)

        self.step_reward = 1

        # 清除舊軌跡線
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

        # 更新車子資訊
        self.car_x_label.setText(f"X: {self.car.x:.2f}")
        self.car_y_label.setText(f"Y: {self.car.y:.2f}")
        self.car_theta_label.setText(f"θ: {self.car.theta:.1f}°")

        # 更新感測器資訊
        sensor = self.car.get_sensor_distances(border_to_segments(self.border_points))
        self.sensor_left_label.setText(f"Left: {sensor[0]:.2f}")
        self.sensor_front_label.setText(f"Front: {sensor[1]:.2f}")
        self.sensor_right_label.setText(f"Right: {sensor[2]:.2f}")

        # 更新軌跡線（加入新座標點）
        x = self.car.x * self.SCALE
        y = -self.car.y * self.SCALE
        # print(x, y)

        # if self.path.isEmpty():
        #    self.path.moveTo(x, y)
        # else:
        self.path.lineTo(x, y)

        if self.trajectory_item:    
            self.trajectory_item.setPath(self.path)
        else:
            # print("=======================================")
            pen = QPen(QColor("white"), 1)  # 半透明黃線
            self.trajectory_item = self.scene.addPath(self.path, pen)
        # print(f"trace point: ({x:.2f}, {y:.2f})")
        # print(f"path element count: {self.path.elementCount()}")
       
        pen = QPen(QColor(0, 0, 255, 70))    # 淡藍色邊框
        brush = QBrush(QColor(0, 0, 255, 30)) # 更淡的藍色填色（或改為透明）

        self.car_item = self.scene.addEllipse(
            self.car.x * self.SCALE - 3 * self.SCALE,
            -self.car.y * self.SCALE - 3 * self.SCALE,
            6 * self.SCALE,
            6 * self.SCALE,
            pen,
            brush
        )
        # self.car_item = self.scene.addEllipse(self.car.x * self.SCALE - 3 * self.SCALE, -self.car.y * self.SCALE - 3 * self.SCALE, 6 * self.SCALE, 6 * self.SCALE, QPen(QColor("blue")))
        rad = math.radians(self.car.theta)
        x2 = self.car.x + math.cos(rad) * 1.0 
        y2 = self.car.y + math.sin(rad) * 1.0 
        pen = QPen(QColor(0, 255, 255, 70), 1)  # 半透明 cyan，alpha=100
        self.car_dir_line = self.scene.addLine(
            self.car.x * self.SCALE,
           -self.car.y * self.SCALE,
            x2 * self.SCALE + 3 * self.SCALE * math.cos(rad),
            -y2 * self.SCALE - 3 * self.SCALE * math.sin(rad),
            pen
        )
        # self.car_dir_line = self.scene.addLine(self.car.x * self.SCALE, -self.car.y * self.SCALE, x2 * self.SCALE + 3 * self.SCALE * math.cos(rad), -y2 * self.SCALE - 3 * self.SCALE * math.sin(rad), QPen(QColor("cyan"), 1))
  
    def start_training(self):
        self.agent = Agent(
            lr=float(self.lr_input.text()),
            discount_factor=float(self.discounted_factor.text()),
            epsilon=float(self.eps_input.text()),
            epsilon_decay=float(self.epsd_input.text())
        )
        self.current_episode = 0
        interval = self.speed_slider.value()  # 單位是毫秒
        self.timer.start(interval)

    def stop_training(self):
        self.timer.stop()
        self.log_decision("🛑 Training manually stopped.")

    def get_reward(self):
        x, y = self.car.x, self.car.y
        radius = 3  # 車子半徑

        gx1, gy1 = self.goal_tl
        gx2, gy2 = self.goal_br
        if gx1 <= x <= gx2 and gy2 <= y <= gy1:
           return 100000000000000, True

        for x1, y1, x2, y2 in border_to_segments(self.border_points):
            if is_circle_near_segment(x, y, radius, x1, y1, x2, y2):
                return -1000, True

        self.step_reward *= 1.2
        return self.step_reward, False

    def update_epoch(self, epoch):
        self.episode_label.setText(f"Epoch: {epoch}")

    def log_decision(self, text):
        self.decision_log.append(text)

    def train_step(self):
        # 1. 取得state
        sensor = self.car.get_sensor_distances(border_to_segments(self.border_points))
        state = self.agent.get_state(sensor)

        # 2. 根據state 選擇action 並更新car 
        action = self.agent.select_action(state)
        angle_choices = [-40, -20, 0, 20, 40]
        # self.car.rotate(angle_choices[action])
        self.car.move_forward(angle_choices[action])

        # 3. 更新state
        next_sensor = self.car.get_sensor_distances(border_to_segments(self.border_points))
        next_state = self.agent.get_state(next_sensor)

        # 4. 計算reward
        reward, done = self.get_reward()

        # 5. 更新 Q-table
        self.agent.update_q_table(state, action, reward, next_state)
        
        # 6. 更新畫面
        self.log_decision(f"EP{self.current_episode} | S:{state} A:{action} R:{reward} -> S':{next_state}")
        self.update_car_graphics()
        
        # 7.
        if done:
            self.agent.decay_epsilon()
            self.current_episode += 1
            self.reset_car()

            if self.current_episode >= int(self.episode_label.text()):
                self.timer.stop()
                self.log_decision("✅ Training complete.")
                self.run_test_episode()  # 加這行！

    def run_test_episode(self):
        self.reset_car()
        self.is_testing = True
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.test_step)
        self.test_timer.start(100)

    def test_step(self):
        sensor = self.car.get_sensor_distances(border_to_segments(self.border_points))
        state = self.agent.get_state(sensor)

        # 完全 greedy 選擇最優動作
        q_values = self.agent.q_table.get(state, [0]*5)
        action = q_values.index(max(q_values))
        angle_choices = [-40, -20, 0, 20, 40]

        # self.car.rotate(angle_choices[action])
        self.car.move_forward(angle_choices[action])

        self.update_car_graphics()

        reward, done = self.get_reward()
        if done:
            self.test_timer.stop()
            self.is_testing = False
            self.log_decision("🧪 Test episode complete.")

