from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTextEdit, QGraphicsView, QGraphicsScene, QFileDialog
)
from PyQt5.QtGui import QPolygonF, QPen, QColor
from PyQt5.QtCore import QPointF
from geometry import parse_track_file


class TrackWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("計算型智慧 作業一 Q-learning")
        self.setGeometry(100, 100, 1000, 600)

        # 建立主畫面 layout（水平切左右）
        main_layout = QHBoxLayout(self)

        # 左側：控制區
        self.control_layout = QVBoxLayout()
        self.init_control_panel()
        main_layout.addLayout(self.control_layout, 1)  # 左邊佔 1 份寬度

        # 右側：畫布區
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        main_layout.addWidget(self.view, 3)  # 右邊佔 3 份寬度

    def init_control_panel(self):
        # 匯入座標檔案
        self.import_btn = QPushButton("Import Track File")
        self.import_btn.clicked.connect(self.import_track)
        self.control_layout.addWidget(self.import_btn)

        # Learning Rate
        self.lr_input = QLineEdit("0.1")
        self.control_layout.addWidget(QLabel("Learning Rate"))
        self.control_layout.addWidget(self.lr_input)

        # Epsilon
        self.eps_input = QLineEdit("0.2")
        self.control_layout.addWidget(QLabel("Epsilon"))
        self.control_layout.addWidget(self.eps_input)

        # 訓練次數
        self.epoch_label = QLabel("Epoch: 0")
        self.control_layout.addWidget(self.epoch_label)
        
        # 決策紀錄
        self.decision_log = QTextEdit()
        self.decision_log.setReadOnly(True)
        
        self.control_layout.addWidget(QLabel("Decision Log"))
        self.control_layout.addWidget(self.decision_log)

    def import_track(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Track File", "", "Text Files (*.txt)")
        if not path:
            return

        # 呼叫 geometry.py 中的解析函式
        start, start_tl, start_br, goal_tl, goal_br, border = parse_track_file(path)
        self.draw_track(start, start_tl, start_br, goal_tl, goal_br, border)

    def draw_track(self, start, start_tl, start_br, goal_tl, goal_br, border_points):
        self.scene.clear()

        # 畫邊界（黑線）
        poly = QPolygonF([QPointF(x * 10, -y * 10) for x, y in border_points])
        self.scene.addPolygon(poly, QPen(QColor("white"), 2))

        # 畫起點（紅點）
        self.scene.addEllipse(start[0] * 10 - 3, -start[1] * 10 - 3, 6, 6, brush=QColor("red"))

        # 起點線
        x1, y1 = start_tl
        x2, y2 = start_br
        self.scene.addLine(x1 * 10, -y1 * 10, x2 * 10, -y2 * 10, QPen(QColor("gray"), 2))

        # 畫終點（綠框）
        x1, y1 = goal_tl
        x2, y2 = goal_br
        goal_poly = QPolygonF([
            QPointF(x1 * 10, -y1 * 10),
            QPointF(x2 * 10, -y1 * 10),
            QPointF(x2 * 10, -y2 * 10),
            QPointF(x1 * 10, -y2 * 10),
        ])
        self.scene.addPolygon(goal_poly, QPen(QColor("green"), 1, style=3))

    def update_epoch(self, epoch):
        self.epoch_label.setText(f"Epoch: {epoch}")

    def log_decision(self, text):
        self.decision_log.append(text)
