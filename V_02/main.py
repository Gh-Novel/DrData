import sys
import os
import cv2
import numpy as np
import pytesseract
import pandas as pd
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, \
    QFileDialog, QAction, QMessageBox, QSplitter, QGroupBox, QFrame, QCheckBox, QLineEdit, QPushButton


class DrData(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DrData")
        self.setGeometry(100, 100, 800, 600)  # Increased the initial width
        self.original_image = None
        self.modified_image = None
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        # Create a QLabel to represent the fixed-size white area
        self.fixed_area_label = QLabel()
        self.fixed_area_label.setStyleSheet("background-color: white; border: 1px solid black;")
        self.fixed_area_label.setFixedSize(595, 842)

        # Sliders for image processing parameters
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)

        self.noise_reduction_slider = QSlider(Qt.Horizontal)
        self.noise_reduction_slider.setMinimum(0)
        self.noise_reduction_slider.setMaximum(100)
        self.noise_reduction_slider.setValue(0)
        self.noise_reduction_slider.setTickInterval(10)
        self.noise_reduction_slider.setTickPosition(QSlider.TicksBelow)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)

        self.brightness_slider.valueChanged.connect(self.process_image)
        self.contrast_slider.valueChanged.connect(self.process_image)
        self.noise_reduction_slider.valueChanged.connect(self.process_image)
        self.threshold_slider.valueChanged.connect(self.process_image)

        
        sliders_layout = QVBoxLayout()

        # Add control set with reduced height
        sliders_layout.addWidget(QLabel("Brightness"))
        sliders_layout.addWidget(self.brightness_slider)
        sliders_layout.addSpacing(0)

        # Add control set with reduced height
        sliders_layout.addWidget(QLabel("Noise Reduction"))
        sliders_layout.addWidget(self.noise_reduction_slider)
        sliders_layout.addSpacing(0)

        # Add control set with reduced height
        sliders_layout.addWidget(QLabel("Contrast"))
        sliders_layout.addWidget(self.contrast_slider)
        sliders_layout.addSpacing(0)

        # Add threshold slider to the layout
        sliders_layout.addWidget(QLabel("Threshold"))
        sliders_layout.addWidget(self.threshold_slider)
        sliders_layout.addSpacing(0)

        # Set a fixed height for each control set
        for i in range(sliders_layout.count()):
            item = sliders_layout.itemAt(i)
            if item and item.widget():
                item.widget().setFixedHeight(20)
        # Create a group box for filter controls
        filters_group = QGroupBox("Filters")
        filters_group.setLayout(sliders_layout)

        # Create a splitter to divide the left side vertically
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(filters_group)

        # Add a new section "L2" in the bottom with a checkbox and label input
        L2_layout = QVBoxLayout()

        # Add checkbox named "Add Label"
        self.add_label_checkbox = QCheckBox("Add Label")
        L2_layout.addWidget(self.add_label_checkbox)

        # Add a line edit for label input initially hidden
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Enter Label")
        self.label_input.hide()
        L2_layout.addWidget(self.label_input)

        # Add a button to add the label
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_label)
        L2_layout.addWidget(add_button)

        # Add a QLabel to display input labels in gray color
        self.labels_display = QLabel()
        self.labels_display.setStyleSheet("color: gray;")
        L2_layout.addWidget(self.labels_display)

        # Set a fixed height for L2
        L2_layout.addStretch(1)
        L2_layout.setContentsMargins(0, 0, 0, 280)
        L2_group = QGroupBox("L2")
        L2_group.setLayout(L2_layout)

        splitter.addWidget(L2_group)
        splitter.addWidget(QFrame())  # You can add any widget here for the second half
        splitter.setSizes([filters_group.sizeHint().height(), 500 - filters_group.sizeHint().height()])

        # Create layout for the entire window
        main_layout = QHBoxLayout()

        # Add the splitter to the left
        main_layout.addWidget(splitter)

        # Add a separator (line) to visually separate the two sides
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        main_layout.addWidget(separator)

        # Add the fixed-size white area on the right
        self.fixed_area_label = QLabel()
        self.fixed_area_label.setFixedSize(500, 600)
        main_layout.addWidget(self.fixed_area_label)
        # Set the main layout for the central widget
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Create a QLabel for the image preview
        self.image_preview_label = QLabel()
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setStyleSheet("background-color: white; border: 1px solid black;")
        self.image_preview_label.setFixedSize(400, 550)  # Set the size of the image preview area

        # Create a layout for centering the image preview label
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.image_preview_label)
        preview_layout.setAlignment(Qt.AlignCenter)

        # Create a frame to contain the centered image preview label
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: gray;")
        preview_frame.setLayout(preview_layout)

        # Add the preview frame to the fixed-size label area
        layout = QVBoxLayout()
        layout.addWidget(preview_frame)
        self.fixed_area_label.setLayout(layout)

        # Set the background color of the fixed-size area on the right to gray
        self.fixed_area_label.setStyleSheet("background-color: gray;")

        # Set the main layout for the central widget
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.create_menu()

        # Connect the checkbox signal to show/hide the label input and save labels
        self.add_label_checkbox.stateChanged.connect(self.toggle_label_input)

    def apply_effects(self):
        if self.original_image is not None:
            # Initialize modified image as a copy of the original image
            self.modified_image = self.original_image.copy()

            # Get the values of the sliders
            brightness_value = self.brightness_slider.value()
            noise_reduction_value = self.noise_reduction_slider.value()
            contrast_value = self.contrast_slider.value()

            # Apply brightness adjustment
            if brightness_value != 0:
                self.modified_image = cv2.addWeighted(
                    self.modified_image,
                    1,
                    np.zeros(self.original_image.shape, self.original_image.dtype),
                    0,
                    brightness_value,
                )

            # Apply noise reduction
            if noise_reduction_value != 0:
                self.modified_image = cv2.fastNlMeansDenoisingColored(
                    self.modified_image, None, noise_reduction_value, 10, 7, 21
                )

            # Apply contrast adjustment
            if contrast_value != 0:
                self.modified_image = cv2.convertScaleAbs(
                    self.modified_image, alpha=1.0, beta=contrast_value
                )

            # Display the modified image
            self.display_image_in_imgbox(self.modified_image)

    def add_label(self):
        label_text = self.label_input.text().strip()
        if label_text:
            labels = self.labels_display.text().split(', ')
            labels.append(label_text)
            self.update_displayed_labels(labels)
            self.label_input.clear()

    def toggle_label_input(self, state):
        if state == Qt.Checked:
            self.label_input.show()
        else:
            self.label_input.hide()

        # If checkbox is unchecked, save labels to label.txt and update displayed labels
        labels = self.label_input.text().split(',')
        labels = [label.strip() for label in labels if label]  # Remove empty labels
        labels_text = ', '.join(labels)

        if labels_text:
            file_path = r"C:\Users\Novel kathor\Desktop\projecte_1\DrData\DrData\label.txt"
            with open(file_path, "w") as file:
                file.write(labels_text)

        self.update_displayed_labels(labels)

    def update_displayed_labels(self, labels):
        labels_text = ', '.join(labels)
        self.labels_display.setText(labels_text)

    def create_menu(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

        extract_action = QAction("Extract", self)
        extract_action.triggered.connect(self. process_folder)
        file_menu.addAction(extract_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close_application)  # Connect to a custom close method
        file_menu.addAction(exit_action)

        # Options Menu
        options_menu = menu_bar.addMenu("Options")

        # About Menu
        about_menu = menu_bar.addMenu("About")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.about_dialog)
        about_menu.addAction(about_action)

    def close_application(self):
        self.close()

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
            self.apply_effects()  # Call apply_effects to apply the changes

    def reduce_noise(self):
        if self.original_image is not None:
            noise_reduction_value = self.noise_reduction_slider.value()
            self.modified_image = cv2.fastNlMeansDenoisingColored(
                self.original_image, None, noise_reduction_value, 10, 7, 21
            )
            self.apply_effects()  # Call apply_effects to apply the changes

    def adjust_contrast(self):
        if self.original_image is not None:
            contrast_value = self.contrast_slider.value()
            self.modified_image = cv2.convertScaleAbs(
                self.original_image, alpha=1.0, beta=contrast_value
            )
            self.apply_effects()  # Call apply_effects to apply the changes

    def adjust_threshold(self):
        if self.modified_image is not None:  # Check if modified_image is not None
            threshold_value = self.threshold_slider.value()
            _, thresholded_image = cv2.threshold(self.modified_image, threshold_value, 255, cv2.THRESH_BINARY)
            self.display_image_in_imgbox(thresholded_image)

    def display_image(self):
        if self.processed_image is not None:
            # Convert processed image to QPixmap
            q_img = QImage(
                self.processed_image.data, self.processed_image.shape[1], self.processed_image.shape[0],
                self.processed_image.strides[0], QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_img)
    
            # Scale the pixmap to fit the size of the image_label
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
    
            # Set the scaled pixmap to the image_label
            self.image_label.setPixmap(scaled_pixmap)
    
            # Scale the pixmap to fit the size of the image_preview_label
            scaled_pixmap_preview = pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio)
    
            # Set the scaled pixmap to the image_preview_label
            self.image_preview_label.setPixmap(scaled_pixmap_preview)


    
    def load_and_display_first_image(self):
        if hasattr(self, 'folder_path'):
            folder_path = self.folder_path
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                first_image_path = os.path.join(folder_path, image_files[0])
                self.image = cv2.imread(first_image_path)
                self.processed_image = self.image.copy()
                self.display_image()



    def resize_image_to_fit(self, image, size):
        height, width, _ = image.shape
        target_width, target_height = size

        # Calculate the aspect ratio of the original image
        aspect_ratio = width / height

        # Calculate the target size while maintaining the aspect ratio
        if width > target_width or height > target_height:
            if aspect_ratio > (target_width / target_height):
                target_height = int(target_width / aspect_ratio)
            else:
                target_width = int(target_height * aspect_ratio)

            # Resize the image
            resized_image = cv2.resize(image, (target_width, target_height))
            return resized_image
        else:
            return image

    def convert_cv_to_pixmap(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        return QPixmap(QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888))
    
    def extract_text_from_image(self):
        if self.processed_image is not None:
            text = pytesseract.image_to_string(self.processed_image)
            print("Extracted Text:", text)
            return text.strip()
        else:
            print("Processed image is None. Cannot extract text.")
            return ""
 
    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if folder_path:
            self.folder_path = folder_path  # Store the folder path for later use
    
            # Load the first image in the folder and display it in the preview window
            self.load_and_display_first_image()
    
    
    def process_folder(self):
        if not hasattr(self, 'folder_path'):
            print("Error: No folder selected.")
            return
    
        folder_path = self.folder_path
    
        if not isinstance(folder_path, str):
            print("Error: Invalid folder path.")
            return
            
        if not os.path.isdir(folder_path):
            print("Error: Path does not exist or is not a directory.")
            return
    
        # Proceed with processing the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
         
        with open('C:/Users/Novel kathor/Desktop/projecte_1/DrData/DrData/temp.txt', 'a') as file:
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                
                # Set self.processed_image for text extraction
                self.processed_image = image
                
                # Extract text from the preprocessed image
                extracted_text = self.extract_text_from_image()
                
                if extracted_text:
                    file.write(f"Data for image: {image_file}\n")
                    file.write(extracted_text + '\n')
                    file.write("--------------------------------------------------\n")
                    print(f"Text extracted and saved for image: {image_file}")
                    # Process and save data to Excel after extracting text from all images
                    self.process_data_and_save_to_excel() 
                else:
                    print(f"No text extracted for image: {image_file}")
        


    def process_data_and_save_to_excel(self):
        data = []
        columns = set()
        processed_images = set()  # Keep track of processed images
        with open('C:/Users/Novel kathor/Desktop/projecte_1/DrData/DrData/temp.txt', 'r') as file:
            current_data = {}
            for line in file:
                line = line.strip()
                if line.startswith("Data for image:"):
                    image_file = line.split("Data for image:")[1].strip()
                    if image_file not in processed_images:  # Process each image only once
                        processed_images.add(image_file)
                        if current_data:
                            data.append(current_data)
                            current_data = {}
                elif line.startswith("-----"):
                    if current_data:
                        data.append(current_data)
                        current_data = {}
                else:
                    key_value = line.split(":")
                    if len(key_value) >= 2:
                        column_name = key_value[0].strip()
                        value = key_value[1].strip()
                        columns.add(column_name)
                        current_data[column_name] = value

            if current_data:
                data.append(current_data)

        columns = list(columns)
        df = pd.DataFrame(data, columns=columns)
        df.to_excel('output.xlsx', index=False)

    def about_dialog(self):
        QMessageBox.about(
            self,
            "About DrData",
            "<b>DrData</b> is a medical data extraction tool developed by TeamXYZ.",
        )

    def process_image(self):
        if self.image is not None:
            brightness_value = self.brightness_slider.value()
            contrast_value = self.contrast_slider.value()
            noise_reduction_value = self.noise_reduction_slider.value()
            threshold_value = self.threshold_slider.value()

            # Adjust brightness
            processed_image = cv2.add(self.image, brightness_value)

            # Adjust contrast
            processed_image = cv2.convertScaleAbs(processed_image, alpha=(100.0 + contrast_value) / 100.0, beta=0)

            # Apply noise reduction
            processed_image = cv2.GaussianBlur(processed_image, (5, 5), noise_reduction_value)

            # Apply thresholding
            _, processed_image = cv2.threshold(processed_image, threshold_value, 255, cv2.THRESH_BINARY)

            self.processed_image = processed_image
            self.display_image()

   # def extract_text(self):
    #    if self.processed_image is not None:
    #       text = pytesseract.image_to_string(self.processed_image)
     #       print("Extracted Text:", text) 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrData()
    window.show()
    sys.exit(app.exec_())
