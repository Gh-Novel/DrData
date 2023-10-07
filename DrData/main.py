import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFileDialog, QMenuBar, QAction, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import pytesseract


class DrData(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Editor")
        self.setGeometry(100, 100, 600, 400)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)  # Set scaledContents to True for proper scaling

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)

        self.noise_reduction_slider = QSlider(Qt.Horizontal)
        self.noise_reduction_slider.setMinimum(0)
        self.noise_reduction_slider.setMaximum(100)
        self.noise_reduction_slider.setValue(0)
        self.noise_reduction_slider.setTickInterval(10)
        self.noise_reduction_slider.setTickPosition(QSlider.TicksBelow)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)

        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        self.noise_reduction_slider.valueChanged.connect(self.reduce_noise)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)

        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(QLabel("Brightness"))
        sliders_layout.addWidget(self.brightness_slider)
        sliders_layout.addWidget(QLabel("Noise Reduction"))
        sliders_layout.addWidget(self.noise_reduction_slider)
        sliders_layout.addWidget(QLabel("Contrast"))
        sliders_layout.addWidget(self.contrast_slider)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.image_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(sliders_layout)
        main_layout.addLayout(preview_layout)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.original_image = None
        self.modified_image = None

        self.create_menu()

    def create_menu(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.browse_image)
        file_menu.addAction(open_action)

        extract_action = QAction("Extract", self)
        extract_action.triggered.connect(self.extract_image)
        file_menu.addAction(extract_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Options Menu
        options_menu = menu_bar.addMenu("Options")

        # About Menu
        about_menu = menu_bar.addMenu("About")
        about_action = QAction("About", self)
        about_menu.addAction(about_action)

    def browse_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.original_image = cv2.imread(file_path)
            self.modified_image = self.original_image.copy()
            self.display_image()

    def display_image(self):
        if self.modified_image is not None:
            height, width, channel = self.modified_image.shape
            bytes_per_line = channel * width
            q_img = QImage(
                self.modified_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)

    def adjust_brightness(self):
        if self.original_image is not None:
            brightness_value = self.brightness_slider.value()
            self.modified_image = cv2.addWeighted(
                self.original_image,
                1,
                np.zeros(self.original_image.shape, self.original_image.dtype),
                0,
                brightness_value,
            )
            self.apply_effects()

    def reduce_noise(self):
        if self.original_image is not None:
            noise_reduction_value = self.noise_reduction_slider.value()
            self.modified_image = cv2.fastNlMeansDenoisingColored(
                self.original_image, None, noise_reduction_value, 10, 7, 21
            )
            self.apply_effects()

    def adjust_contrast(self):
        if self.original_image is not None:
            contrast_value = self.contrast_slider.value()
            self.modified_image = cv2.convertScaleAbs(
                self.original_image, alpha=1.0, beta=contrast_value
            )
            self.apply_effects()

    def apply_effects(self):
        if self.original_image is not None:
            # Apply the cumulative effects to the original image
            self.modified_image = self.original_image.copy()
            brightness_value = self.brightness_slider.value()
            noise_reduction_value = self.noise_reduction_slider.value()
            contrast_value = self.contrast_slider.value()

            if brightness_value != 0:
                self.modified_image = cv2.addWeighted(
                    self.modified_image,
                    1,
                    np.zeros(self.modified_image.shape, self.modified_image.dtype),
                    0,
                    brightness_value,
                )

            if noise_reduction_value != 0:
                self.modified_image = cv2.fastNlMeansDenoisingColored(
                    self.modified_image, None, noise_reduction_value, 10, 7, 21
                )

            if contrast_value != 0:
                self.modified_image = cv2.convertScaleAbs(
                    self.modified_image, alpha=1.0, beta=contrast_value
                )

            self.display_image()

    def extract_image(self):
        if self.modified_image is not None:
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)
            extracted_data = pytesseract.image_to_string(gray_image)
            file_path = r"C:\Users\my pc\Desktop\project\temp.txt"
            with open(file_path, "w") as file:
                file.write(extracted_data)
            QMessageBox.information(
                self, "Extraction Successful", "Text data extracted and saved to file."
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = DrData()
    editor.show()
    sys.exit(app.exec_())
