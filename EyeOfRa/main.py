import os.path
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import QFile
from PySide6 import QtCore
from PySide6 import QtWidgets
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import Slot
from PySide6 import QtGui
import pathlib
from EyeOfRa.resources.ui.ui_mainwindow import Ui_MainWindow
from EyeOfRa.scripts.frontend.model_setup import Model_Setup
from EyeOfRa.scripts.backend.image_converter import convert as convert_image
from EyeOfRa.scripts.frontend.loaded_model_widget import Loaded_Model_Widget


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.components = []
        self.run_options = {"model_path": None,
                            "pre_classification_image": None}



        #  Add model_setup
        self.model_setup = Model_Setup()
        self.model_setup.setParent(self.ui.frame_left)
        self.model_setup.setMaximumHeight(100)
        layout: QtWidgets.QLayout = self.ui.frame_left.layout()
        layout.addWidget(self.model_setup)


        # Set images
        im = "resources/development/example_images/placeholder.tif"
        self.set_pre_class_image(im)



        # button logic
        self.model_setup.ui.button_load.clicked.connect(self.load_model_pressed)
        self.ui.actionLoad_Model.triggered.connect(self.load_model_pressed)
        self.ui.btn_upload_image.clicked.connect(self.load_image_button_pressed)

    def set_pre_class_image(self, path):
        temp_path = "resources/temp"
        if not os.path.exists(path):
            self.print_to_debug(f"Path: '{path}' does not exist", mode="error")

        path = pathlib.Path(path)
        export_name = None
        if path.suffix not in [".jpg", ".png"]:
            export_name = f"{temp_path}/{path.stem}_conv.png"
            convert_image(path, export_name)
            path = export_name

        print(f"Image path = {path}")
        label = self.ui.label_image
        pixmap = QtGui.QPixmap(path)
        label.setPixmap(pixmap)
        if export_name:
            os.remove(export_name)

    def print_to_debug(self, string: str, mode="debug"):
        debug = True
        browser = self.ui.textBrowser_debug
        if mode == "debug" and debug:
            browser.setTextColor("black")
            browser.append(f"[DEBUG]\t{string}")
            browser.setTextColor("black")

        if mode == "info":
            browser.setTextColor("black")
            browser.append(f"[INFO]\t{string}")
            browser.setTextColor("black")

        if mode == "warning":
            browser.setTextColor("orange")
            browser.append(f"[WARNING]\t{string}")
            browser.setTextColor("black")

        if mode == "error":
            browser.setTextColor("red")
            browser.append(f"[ERROR]\t{string}")
            browser.setTextColor("black")

    def file_dialog(self, path, extentions=""):
        filename = QFileDialog.getOpenFileName(self,
                                               ("Open Image"), path,
                                               extentions)
        return filename

    @Slot()
    def load_model_pressed(self):
        temp_path = "resources/model_history"
        filename, _ = self.file_dialog(temp_path, "*h5")
        if filename:
            self.print_to_debug(f"Added: {filename}", mode="debug")
            self.run_options["model"] = filename
            self.model_setup.hide()
            self.loaded_model = Loaded_Model_Widget(filename=filename)
            self.loaded_model.setParent(self.ui.frame_left)
            self.loaded_model.ui.btn_close_model.clicked.connect(self.close_loaded_model_pressed)
            self.ui.frame_left.layout().addWidget(self.loaded_model, 0)
            self.loaded_model.show()

    @Slot()
    def close_loaded_model_pressed(self):
        if self.loaded_model:
            self.print_to_debug("Closing loaded model")
            self.loaded_model.hide()
            self.model_setup.show()
            self.run_options["model"] = None
        else:
            self.print_to_debug("Nothing to hide", mode="warning")

    @Slot()
    def load_image_button_pressed(self):
        path, _ = self.file_dialog("resources//development/example_images", extentions="*.tif")
        if path:
            self.set_pre_class_image(path)
        else:
            self.print_to_debug("Canceled action: Upload image", mode="debug")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    # Resize window
    size: QtCore.QRect = QGuiApplication.primaryScreen().availableGeometry()
    window.resize(size.width(), size.height())

    window.show()

    sys.exit(app.exec_())
