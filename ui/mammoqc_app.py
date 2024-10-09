import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget,
                             QHBoxLayout, QListWidget, QLabel, QSplitter, QSizePolicy, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QImage, QResizeEvent, QPainter, QPen, QColor
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path to import vindr_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.image import dicom_to_image
import inference.view_orientation_clasiffication_inference as view_orientation_classification
import inference.nipple_detection_inference as nipple_detection
import inference.pectoral_muscle_segmentation_inference as pectoral_muscle_segmentation
import processing.pnd as pnd
import skimage.transform

class ScalableImageLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignCenter)
        self.original_pixmap = None

    def setPixmap(self, pixmap: QPixmap) -> None:
        self.original_pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MammoQC")
        self.setGeometry(100, 100, 1200, 600)
        
        main_layout = QHBoxLayout()
        
        # Left panel (file list and buttons)
        left_layout = QVBoxLayout()
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Drag and drop DICOM files here...")
        self.text_edit.setAcceptDrops(True)
        self.text_edit.dragEnterEvent = self.dragEnterEvent
        self.text_edit.dropEvent = self.dropEvent
        left_layout.addWidget(self.text_edit)
        
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self.process_files)
        left_layout.addWidget(self.process_button)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.display_image)
        left_layout.addWidget(self.file_list)
        
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_image)
        left_layout.addWidget(self.analyze_button)
        
        left_container = QWidget()
        left_container.setLayout(left_layout)
        
        # Center panel (image display)
        self.image_label = ScalableImageLabel()
        self.image_label.setText("No Image")
        
        # Right panel (analysis info)
        right_layout = QVBoxLayout()
        
        info_group = QGroupBox("Analysis Information")
        info_layout = QFormLayout()
        
        self.orientation_label = QLabel("N/A")
        self.view_label = QLabel("N/A")
        self.pnd_label = QLabel("N/A")
        
        info_layout.addRow("Orientation:", self.orientation_label)
        info_layout.addRow("View:", self.view_label)
        info_layout.addRow("PND:", self.pnd_label)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        right_layout.addStretch(1)  # Add stretch to push the group box to the top
        
        right_container = QWidget()
        right_container.setLayout(right_layout)
        
        # Main layout setup
        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.addWidget(left_container)
        splitter1.addWidget(self.image_label)
        
        splitter2 = QSplitter(Qt.Horizontal)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(right_container)
        
        splitter1.setSizes([300, 600])
        splitter2.setSizes([900, 300])
        
        main_layout.addWidget(splitter2)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.file_paths = []
        self.processed_images = {}
        self.pixel_spacings = {}
        self.analysis_results = {}  
        self.current_image = None
        self.current_pixel_spacing = None
        self.current_filename = None
        
        self.initialized_models = False
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.dcm', '.dicom')):
                self.file_paths.append(file_path)
                self.text_edit.append(file_path)
            else:
                self.text_edit.append(f"Ignored non-DICOM file: {file_path}")
    
    def process_files(self):
        if not self.file_paths:
            self.text_edit.append("No DICOM files to process.")
            return
        
        self.text_edit.append("Processing files:")
        self.file_list.clear()
        for file_path in self.file_paths:
            filename = os.path.basename(file_path)
            self.processed_images[filename], pixel_spacing = dicom_to_image(file_path)
            self.pixel_spacings[filename] = pixel_spacing  # Store pixel spacing for each image
            self.text_edit.append(f"Processed: {filename}")
            self.file_list.addItem(filename)
        
        self.file_paths.clear()
        self.text_edit.append("Processing complete.")
    
    def display_image(self, item):
        filename = item.text()
        if filename in self.processed_images:
            self.current_image = self.processed_images[filename]
            self.current_pixel_spacing = self.pixel_spacings[filename]
            self.current_filename = filename
            qimage = self.pil_image_to_qimage(self.current_image)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)
            
            # Display cached analysis results if available
            if filename in self.analysis_results:
                self.display_analysis_results(filename)
            else:
                self.clear_analysis_results()
        else:
            self.image_label.clear()
            self.image_label.setText("Image not found")
            self.clear_analysis_results()
    
    def display_analysis_results(self, filename):
        results = self.analysis_results[filename]
        self.orientation_label.setText(results['orientation'])
        self.view_label.setText(results['view'])
        self.pnd_label.setText(results['pnd'])
        
        # Create and display the image with overlaid mask and annotations
        combined_pixmap = self.create_combined_image(results)
        self.image_label.setPixmap(combined_pixmap)

    def clear_analysis_results(self):
        self.orientation_label.setText("N/A")
        self.view_label.setText("N/A")
        self.pnd_label.setText("N/A")
        
    def analyze_image(self):
        if self.current_image is None or self.current_filename is None:
            self.text_edit.append("No image selected for analysis.")
            return
        
        # Check if analysis results are already cached
        if self.current_filename in self.analysis_results:
            self.text_edit.append("Using cached analysis results.")
            self.display_analysis_results(self.current_filename)
            return
        
        if not self.initialized_models:
            self.initialize_models()
        
        # Perform analysis
        view, orientation, _, _ = self.prediction_view_orientation()
        nipple_bbox = self.prediction_nipple()
        pectoral_muscle_mask = self.prediction_pectoral_muscle(self.device)
        
        # Compute PND
        _, _, x, y, slope, intercept, x_nipple, y_nipple, perp_slope, perp_intercept, x_intersect, y_intersect, distance_pixels = pnd.compute_pnd(
            np.array(pectoral_muscle_mask), nipple_bbox[0], nipple_bbox[1], nipple_bbox[2], nipple_bbox[3], orientation, view
        )
        
        # Convert PND to mm if possible
        if self.current_pixel_spacing:
            distance_mm = distance_pixels * self.current_pixel_spacing[0]
            pnd_text = f"{distance_mm:.2f} mm"
        else:
            pnd_text = f"{distance_pixels:.2f} pixels (pixel spacing unknown)"
        
        # Create analyzed image
        # analyzed_pixmap = self.create_analyzed_image(view, orientation, nipple_bbox, x_nipple, y_nipple, slope, intercept, perp_slope, perp_intercept)
        
        # Create analyzed image with overlaid mask
        combined_pixmap = self.create_combined_image({
            'view': view,
            'orientation': orientation,
            'pectoral_muscle_mask': pectoral_muscle_mask,
            'nipple_bbox': nipple_bbox,
            'x_nipple': x_nipple,
            'y_nipple': y_nipple,
            'slope': slope,
            'intercept': intercept,
            'perp_slope': perp_slope,
            'perp_intercept': perp_intercept
        })
        
        # Store analysis results
        self.analysis_results[self.current_filename] = {
            'view': view,
            'orientation': orientation,
            'pnd': pnd_text,
            'pectoral_muscle_mask': pectoral_muscle_mask,
            'nipple_bbox': nipple_bbox,
            'x_nipple': x_nipple,
            'y_nipple': y_nipple,
            'slope': slope,
            'intercept': intercept,
            'perp_slope': perp_slope,
            'perp_intercept': perp_intercept
        }
        
        # Display results
        self.display_analysis_results(self.current_filename)
        
        self.text_edit.append(f"Image analysis completed. View: {view}, Orientation: {orientation}, PND: {pnd_text}")

    def create_combined_image(self, results):
        # Convert the original image to QPixmap
        qimage = self.pil_image_to_qimage(self.current_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Create a transparent overlay for the mask
        mask_overlay = QPixmap(pixmap.size())
        mask_overlay.fill(Qt.transparent)
        
        painter = QPainter(mask_overlay)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the pectoral muscle mask with transparency
        mask_image = results['pectoral_muscle_mask']
        mask_qimage = self.pil_image_to_qimage(mask_image)
        painter.setOpacity(0.3)  # Adjust transparency here (0.0 to 1.0)
        painter.drawImage(0, 0, mask_qimage)
        
        # Draw annotations
        painter.setOpacity(1.0)
        view = results['view']
        orientation = results['orientation']
        nipple_bbox = results['nipple_bbox']
        x_nipple, y_nipple = results['x_nipple'], results['y_nipple']
        slope, intercept = results['slope'], results['intercept']
        perp_slope, perp_intercept = results['perp_slope'], results['perp_intercept']
        
        # Draw pectoral muscle line
        pen = QPen(Qt.blue)
        pen.setWidth(10)
        painter.setPen(pen)
        if view.upper() == "CC":
            if orientation.upper() == "LEFT":
                painter.drawLine(0, 0, 0, pixmap.height())
            elif orientation.upper() == "RIGHT":
                painter.drawLine(pixmap.width() - 1, 0, pixmap.width() - 1, pixmap.height())
        else:  # MLO view
            y1 = int(slope * 0 + intercept)
            y2 = int(slope * pixmap.width() + intercept)
            painter.drawLine(0, y1, pixmap.width(), y2)
        
        # Draw perpendicular line
        pen.setColor(Qt.green)
        painter.setPen(pen)
        if view.upper() == "CC":
            painter.drawLine(0, int(y_nipple), pixmap.width(), int(y_nipple))
        else:  # MLO view
            x1 = int((0 - perp_intercept) / perp_slope)
            x2 = int((pixmap.height() - perp_intercept) / perp_slope)
            painter.drawLine(x1, 0, x2, pixmap.height())
        
        # Draw nipple location
        pen.setColor(Qt.red)
        painter.setPen(pen)
        painter.setBrush(Qt.red)
        nipple_radius = 10
        painter.drawEllipse(int(x_nipple) - nipple_radius, int(y_nipple) - nipple_radius, nipple_radius * 2, nipple_radius * 2)
        
        # Draw nipple bounding box
        pen.setColor(Qt.yellow)
        pen.setWidth(10)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRect(nipple_bbox[0], nipple_bbox[1], nipple_bbox[2]-nipple_bbox[0], nipple_bbox[3]-nipple_bbox[1]))
        
        painter.end()
        
        # Combine the original image with the overlay
        result_pixmap = QPixmap(pixmap)
        painter = QPainter(result_pixmap)
        painter.drawPixmap(0, 0, mask_overlay)
        painter.end()
        
        return result_pixmap
    
    def initialize_models(self):
        self.text_edit.append("Initializing models...")
        self.model_view_orientation_classifier = view_orientation_classification.load_model('checkpoints/res2next-mammography-epoch=09-val_loss=0.00.ckpt')
        self.model_nipple_detection = nipple_detection.load_model(r'D:\Code\MammoQC\runs\detect\train4\weights\best.pt')
        self.model_pectoral_muscle_segmentation, self.device = pectoral_muscle_segmentation.load_model('checkpoints/pectoral-segmentation-unet-512-epoch=06-val_dice_coeff=0.97.ckpt')
        self.initialized_models = True

    def create_analyzed_image(self, view, orientation, nipple_bbox, x_nipple, y_nipple, slope, intercept, perp_slope, perp_intercept):
        qimage = self.pil_image_to_qimage(self.current_image)
        pixmap = QPixmap.fromImage(qimage)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw pectoral muscle line
        pen = QPen(Qt.blue)
        pen.setWidth(30)
        painter.setPen(pen)
        if view.upper() == "CC":
            if orientation.upper() == "LEFT":
                painter.drawLine(0, 0, 0, pixmap.height())
            elif orientation.upper() == "RIGHT":
                painter.drawLine(pixmap.width() - 1, 0, pixmap.width() - 1, pixmap.height())
        else:  # MLO view
            y1 = int(slope * 0 + intercept)
            y2 = int(slope * pixmap.width() + intercept)
            painter.drawLine(0, y1, pixmap.width(), y2)
        
        # Draw perpendicular line
        pen.setColor(Qt.green)
        painter.setPen(pen)
        if view.upper() == "CC":
            painter.drawLine(0, int(y_nipple), pixmap.width(), int(y_nipple))
        else:  # MLO view
            x1 = int((0 - perp_intercept) / perp_slope)
            x2 = int((pixmap.height() - perp_intercept) / perp_slope)
            painter.drawLine(x1, 0, x2, pixmap.height())
        
        # Draw nipple location
        pen.setColor(Qt.red)
        painter.setPen(pen)
        painter.setBrush(Qt.red)
        nipple_radius = 50
        painter.drawEllipse(int(x_nipple) - nipple_radius, int(y_nipple) - nipple_radius, nipple_radius * 2, nipple_radius * 2)
        
        # Draw nipple bounding box
        pen.setColor(Qt.yellow)
        pen.setWidth(30)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRect(nipple_bbox[0], nipple_bbox[1], nipple_bbox[2]-nipple_bbox[0], nipple_bbox[3]-nipple_bbox[1]))
        
        painter.end()
        return pixmap
      
    def prediction_pectoral_muscle(self, device):
        processed_image = pectoral_muscle_segmentation.preprocess_image_from_numpy(self.current_image, device)
        pectoral_muscle_segmentation_mask = pectoral_muscle_segmentation.predict(self.model_pectoral_muscle_segmentation, processed_image)
        
        # Convert tensor to numpy array
        mask_np = pectoral_muscle_segmentation_mask.cpu().numpy().squeeze()
        mask_np = skimage.transform.resize(mask_np, self.current_image.size, order=0, preserve_range=True, anti_aliasing=False)
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    
        # Scale the mask back to the original image size
        # original_size = self.current_image.size
        # mask_image = mask_image.resize(original_size, Image.BICUBIC)
        return mask_image

    def prediction_nipple(self):
        img_tensor = nipple_detection.preprocess_model_input(self.current_image)
    
        results = self.model_nipple_detection(img_tensor)
        bbox_640 = nipple_detection.get_nipple_bbox(results)
        bbox_orig = nipple_detection.scale_bbox(bbox_640, self.current_image.size)
        return bbox_orig

    def prediction_view_orientation(self):
        image_tensor = view_orientation_classification.preprocess_image_from_numpy(self.current_image)
        view, orientation, view_probs, orientation_probs = view_orientation_classification.predict(self.model_view_orientation_classifier, image_tensor)
        return view,orientation,view_probs,orientation_probs
    
    def pil_image_to_qimage(self, pil_image):
        if pil_image.mode == "RGB":
            r, g, b = pil_image.split()
            pil_image = Image.merge("RGB", (b, g, r))
        elif pil_image.mode == "RGBA":
            r, g, b, a = pil_image.split()
            pil_image = Image.merge("RGBA", (b, g, r, a))
        elif pil_image.mode == "L":
            pil_image = pil_image.convert("RGBA")
        
        img_data = pil_image.tobytes("raw", pil_image.mode)
        qimage = QImage(img_data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
        return qimage

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())