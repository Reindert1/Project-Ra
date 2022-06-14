from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from EyeOfRa.resources.ui.ui_loadedmodelwidget import Ui_Form
from PySide6.QtCore import Slot


class Loaded_Model_Widget(QWidget):
    def __init__(self, filename):
        super(Loaded_Model_Widget, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.model_path = None
        self.ui.model_path.setText(filename)
