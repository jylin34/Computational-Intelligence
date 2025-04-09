import sys
from PyQt5.QtWidgets import QApplication
from gui import TrackWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrackWindow()
    window.show()
    sys.exit(app.exec_())
