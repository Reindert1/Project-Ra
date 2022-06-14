from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from EyeOfRa.resources.ui.ui_modelsetup import Ui_Form
from PySide6.QtCore import Slot


class Model_Setup(QWidget):

    def __init__(self):
        super(Model_Setup, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.model_path = None

