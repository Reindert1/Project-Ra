import os.path
import os
import subprocess
import sys
import threading
import typing
import random
import binascii
import time

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QHeaderView
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
from EyeOfRa.scripts.pipeline.pipeline_manager import PipeLineManager
from EyeOfRa.scripts.frontend.loaded_classification_image import Loaded_Classification_Image
# Check if darkmode is enabled
import darkdetect


def in_table(table: QtWidgets.QTableWidget, text):
    """Checks if text is in table"""
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if item:
                if item.text() == text:
                    return True, row, column

    return False, None, None


def remove_item_from_table_by_column(table, column, text):
    table: QtWidgets.QTableWidget
    for row in range(table.rowCount()):
        item = table.item(row, column)
        if item:
            if text == item.text():
                table.removeRow(row)
                table.resizeColumnsToContents()
                return 1
    return 0


def default_run_options(train_data, classifiers, algorithm, gaussian_layers, early_stopping,
                        max_epochs, segment, results_dir, dataset_dir) -> {}:
    options = {}
    options[
        "datadir"] = ""
    options["train_data"] = train_data
    options["classifiers"] = classifiers
    options["algorithms"] = [algorithm]
    options["gaussian_layers"] = gaussian_layers
    options["window_size"] = (10, 10)
    options["early_stopping"] = early_stopping
    options["max_epochs"] = max_epochs
    options["segment"] = {"full": segment}
    options["results_dir"] = results_dir
    options["dataset_dir"] = dataset_dir
    return options


def openImage(path):
    imageViewerFromCommandLine = {'linux': 'xdg-open',
                                  'win32': 'explorer',
                                  'darwin': 'open'}[sys.platform]
    subprocess.run([imageViewerFromCommandLine, path])


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.components = {}
        # Run params for parsing to snakemake interface
        self.run_options = {"model_path": None,
                            "pre_classification_image": None}

        self.run_files = {"training": None,
                          "masks": {},
                          "classification": None,
                          "results_path": None,
                          "dataset_dir": None}
        self.mode = None

        #  Add model_setup
        self.model_setup = Model_Setup()
        self.model_setup.setParent(self.ui.frame_left)
        self.model_setup.setMaximumHeight(100)
        layout: QtWidgets.QLayout = self.ui.frame_left.layout()
        layout.addWidget(self.model_setup)

        # Tab management
        self.tab_manager("close-all")
        self.init_mask_table()

        # Set images
        im = "resources/development/example_images/placeholder.tif"
        for widget in [self.ui.label_image,
                       self.ui.label_image_training_classification,
                       self.ui.label_image_training_training]:
            self.set_image_to_label(im, label=widget)

        # button logic
        self.model_setup.ui.button_load.setEnabled(False)
        self.model_setup.ui.button_load.clicked.connect(self.load_model_clicked)
        self.ui.actionLoad_Model.triggered.connect(self.load_model_clicked)
        self.ui.btn_upload_image.clicked.connect(self.load_image_button_pressed)
        self.model_setup.ui.button_new.clicked.connect(self.new_model_clicked)
        self.ui.btn_upload_image_training_training.clicked.connect(self.set_training_training_image)
        self.ui.pushButton_training_add_new_mask.clicked.connect(self.add_mask_image_to_table)
        self.ui.pushButton_training_remove_all_masks.clicked.connect(self.remove_all_mask_clicked)
        self.ui.btn_upload_image_classification.clicked.connect(
            self.btn_upload_image_classification_clicked)
        self.ui.pushButton_model_export.clicked.connect(self.select_model_export_path)
        self.ui.pushButton_metrics_export.clicked.connect(self.select_metrics_export_path)
        self.ui.pushButton_start_training.clicked.connect(self.start_training_pressed)

    @Slot()
    def start_training_pressed(self):
        algorithm = self.ui.comboBox_algorithm.currentText()

        training = self.run_files["training"]
        masks = self.run_files["masks"]
        classification = self.run_files["classification"]
        dataset_dir = self.run_files["dataset_dir"]
        results_path = self.run_files["results_path"]
        early_stopping = self.ui.checkBox_early_stopping.isChecked()
        max_epochs = self.ui.spinBox_max_epochs.value()
        options = default_run_options(training, masks, algorithm, 6, early_stopping, max_epochs, classification, results_path, dataset_dir)
        run_manager = PipeLineManager()
        for key, val in options.items():
            print(key, val)

        run_manager.yaml_constructor("scripts/pipeline/Ipy/config/config.yaml")
        run_manager.options = options
        self.print_to_debug("Starting Thread for training")
        x = threading.Thread(target=self.wrapped_worker, args=(run_manager,))
        x.start()


    def run_snakemake(self, manager):
        manager.set_default_options()
        # log = manager.run_full(wd="scripts/pipeline/Ipy")
        time.sleep(1)
        log = "Finished Training"
        return log

    def wrapped_worker(self, manager):
        log = self.run_snakemake(manager)
        self.thread_finished(log)

    def thread_finished(self, log):
        self.print_to_debug(log, mode="warning")
        path = "/Users/sanderbouwman/School/Thema11/Themaopdracht/Project-Ra/Project-Ra/EyeOfRa/scripts/pipeline/warehouse/results/images/overlayed/full_NN_overlay.tif"

        openImage(path)

    @Slot()
    def select_metrics_export_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory", "resources/development")

        if path:
            self.run_files["results_path"] = path
            if len(path) > 45:
                path = f"..{path[len(path) - 45:]}"
            self.ui.pushButton_metrics_export.setText(str(path))

    @Slot()
    def select_model_export_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory", "resources/development")

        if path:
            self.run_files["dataset_dir"] = path
            if len(path) > 45:
                path = f"..{path[len(path) - 45:]}"
            self.ui.pushButton_model_export.setText(str(path))


    @Slot()
    def set_training_training_image(self):

        succes, path = self.prompt_and_parse_image("resources/development/example_images",
                                                   extentions="*.tif "
                                                              "*.tiff",
                                                   parse_label=self.ui.label_image_training_training)
        if succes:
            self.run_files["training"] = path

    @Slot()
    def btn_upload_image_classification_clicked(self):
        succes, path = self.prompt_and_parse_image("resources/development/example_images",
                                                   extentions="*.tif "
                                                              "*.tiff",
                                                   parse_label=self.ui.label_image_training_classification)
        if succes:
            self.run_files["classification"] = path

    def prompt_and_parse_image(self, starting_path, extentions, parse_label):
        # Add all parse to this implementations
        path, _ = self.file_dialog(starting_path, extentions=extentions)
        if path:
            self.set_image_to_label(path, label=parse_label)
            return True, path

        else:
            return False, path

    def remove_image_from_label(self, label):
        im = "resources/development/example_images/placeholder.tif"
        self.set_image_to_label(im, label)

    def set_image_to_label(self, image_path, label, max_size=(500, 500)):
        temp_path = "resources/temp"
        if not os.path.exists(image_path):
            self.print_to_debug(f"Path: '{image_path}' does not exist", mode="error")

        path = pathlib.Path(image_path)
        export_name = None

        if path.suffix not in [".jpg", ".png"]:
            random_hex_name = binascii.b2a_hex(os.urandom(15)).decode("utf-8")
            export_name = f"{temp_path}/{random_hex_name}.png"
            convert_image(path, export_name)
            path = export_name

        pixmap = QtGui.QPixmap(path)
        if pixmap.height() > max_size[1] or pixmap.width() > max_size[0]:
            pixmap = pixmap.scaled(max_size[0], max_size[1], QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        if export_name:
            os.remove(export_name)

    def print_to_debug(self, string: str, mode="debug"):
        debug = True
        browser = self.ui.textBrowser_debug
        blackcolor = "black" if darkdetect.isLight() else "white"
        darkdetect.isDark()
        if mode == "debug" and debug:
            browser.setTextColor(blackcolor)
            browser.append(f"[DEBUG]\t{string}")
            browser.setTextColor(blackcolor)

        if mode == "info":
            browser.setTextColor(blackcolor)
            browser.append(f"[INFO]\t{string}")
            browser.setTextColor("black")

        if mode == "warning":
            browser.setTextColor("orange")
            browser.append(f"[WARNING]\t{string}")
            browser.setTextColor(blackcolor)

        if mode == "error":
            browser.setTextColor("red")
            browser.append(f"[ERROR]\t{string}")
            browser.setTextColor(blackcolor)

    def file_dialog(self, path, extentions=""):
        filename = QFileDialog.getOpenFileName(self,
                                               ("Open Image"), path,
                                               extentions)
        return filename

    @Slot()
    def new_model_clicked(self):
        self.mode = "train"
        self.tab_manager("train")

    @Slot()
    def load_model_clicked(self):
        self.mode = "train"
        temp_path = "resources/model_history"
        filename, _ = self.file_dialog(temp_path, "*h5 *sav")
        if filename:
            self.run_options["model"] = filename
            self.model_setup.hide()
            self.loaded_model = Loaded_Model_Widget(filename=filename)
            self.loaded_model.setParent(self.ui.frame_left)
            self.loaded_model.ui.btn_close_model.clicked.connect(self.close_loaded_model_pressed)
            self.ui.frame_left.layout().addWidget(self.loaded_model, 0)
            self.loaded_model.show()
            self.tab_manager("classify")

    @Slot()
    def close_loaded_model_pressed(self):
        self.mode = None
        if self.loaded_model:
            self.loaded_model.hide()
            self.model_setup.show()
            self.run_options["model"] = None
            self.tab_manager("close-all")
        else:
            self.print_to_debug("Nothing to hide", mode="warning")

    @Slot()
    def load_image_button_pressed(self):
        path, _ = self.file_dialog("resources//development/example_images", extentions="*.tif")
        if path:
            # self.set_pre_class_image(path)
            self.set_image_to_label(path, label=self.ui.label_image)

            self.run_files["classification"] = path

        else:
            self.print_to_debug("Canceled action: Upload image", mode="debug")

    def tab_manager(self, tab_clicked=None):
        self.print_to_debug(tab_clicked, mode="debug")

        if tab_clicked == "classify":
            self.ui.tabWidget_right_menu.setTabEnabled(1, True)
            self.ui.tabWidget_right_menu.setTabVisible(1, True)

            self.ui.tabWidget_right_menu.setTabEnabled(2, False)
            self.ui.tabWidget_right_menu.setTabVisible(2, False)
            self.ui.tabWidget_right_menu.setCurrentIndex(1)

        elif tab_clicked == "train":
            self.ui.tabWidget_right_menu.setTabEnabled(1, False)
            self.ui.tabWidget_right_menu.setTabVisible(1, False)
            self.ui.tabWidget_right_menu.setTabEnabled(2, True)
            self.ui.tabWidget_right_menu.setTabVisible(2, True)
            self.ui.tabWidget_right_menu.setCurrentIndex(2)

        else:
            self.ui.tabWidget_right_menu.setTabEnabled(1, False)
            self.ui.tabWidget_right_menu.setTabVisible(1, False)
            self.ui.tabWidget_right_menu.setTabEnabled(2, False)
            self.ui.tabWidget_right_menu.setTabVisible(2, False)
            self.ui.tabWidget_right_menu.setCurrentIndex(0)

    def init_mask_table(self):
        self.components["mask_table_buttons"] = []
        table = self.ui.tableWidget_masks
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Name', 'Path', 'Remove'])


    @Slot()
    def remove_item_from_table_widget(self):
        btn_data: {} = self.get_row_from_button(self.sender())
        table = self.ui.tableWidget_masks

        name = btn_data["name"]
        if remove_item_from_table_by_column(table, 0, name):
            for i, button in enumerate(self.components["mask_table_buttons"]):
                if button["name"] == name:
                    self.components["mask_table_buttons"].pop(i)
                    self.run_files["masks"].pop(name)

    def get_row_from_button(self, sender):
        for row in self.components["mask_table_buttons"]:
            if sender == row["button"]:
                return row

    @Slot()
    def remove_all_mask_clicked(self):
        table = self.ui.tableWidget_masks
        table.setRowCount(0)
        self.components["mask_table_buttons"] = []
        self.run_files["masks"].clear()

        table.resizeColumnsToContents()

    @Slot()
    def add_mask_image_to_table(self):
        table = self.ui.tableWidget_masks

        path, _ = self.file_dialog("resources//development/example_images", extentions="*.tif")
        name = self.prompt_line_input("Please enter a name for masking", "Provide Mask Name")

        name_in_table, _, _ = in_table(table=table, text=name)
        path_in_table, _, _ = in_table(table=table, text=path)
        if name_in_table or path_in_table:
            self.print_to_debug(f"Name or Path already in table. Name/Path should be unique",
                                mode="warning")
            return

        if path and name:
            rowPosition = table.rowCount()
            table.insertRow(rowPosition)
            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.clicked.connect(self.remove_item_from_table_widget)
            self.components["mask_table_buttons"].append(
                {"name": name, "path": path, "button": remove_btn})
            self.run_files["masks"][name] = path
            items = [QtWidgets.QTableWidgetItem(name), QtWidgets.QTableWidgetItem(path)]
            for i, item in enumerate(items):
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(rowPosition, i, item)

            table.setCellWidget(rowPosition, 2, remove_btn)
            table.resizeColumnsToContents()

    def prompt_line_input(self, label, window_name="Input Dialog"):
        ec = QtWidgets.QLineEdit.EchoMode.Normal
        text, ok = QtWidgets.QInputDialog.getText(self, window_name,
                                                  label, echo=ec)

        if ok:
            return text


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    # Resize window
    size: QtCore.QRect = QGuiApplication.primaryScreen().availableGeometry()
    window.resize(size.width(), size.height())

    window.show()

    sys.exit(app.exec_())
