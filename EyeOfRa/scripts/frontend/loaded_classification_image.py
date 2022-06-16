from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from EyeOfRa.resources.ui.ui_loadedclassified import Ui_Classification_image
from PySide6.QtCore import Slot


class Loaded_Classification_Image(QWidget):
    def __init__(self):
        super(Loaded_Classification_Image, self).__init__()
        self.ui = Ui_Classification_image()
        self.ui.setupUi(self)

