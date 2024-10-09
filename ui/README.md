# MammoQC Application

## Overview
MammoQC is a PyQt5-based desktop application for analyzing and quality-checking mammography images. It provides a user-friendly interface for processing DICOM files, visualizing mammograms, and performing automated analysis of breast imaging features.

## Features
- DICOM file loading and processing
- Image viewing with zoom and pan capabilities
- Automated analysis of mammograms, including:
  - View and orientation classification
  - Nipple detection
  - Pectoral muscle segmentation
  - Posterior nipple line (PNL) measurement
- Results caching for quick re-analysis
- Interactive display of analysis results overlaid on the mammogram

## Requirements
- Python 3.x
- PyQt5
- PIL (Pillow)
- NumPy
- Matplotlib
- scikit-image
- CUDA-compatible GPU (recommended for faster processing)

## Installation
1. Clone the repository or download the source code.
2. Install the required packages:
   ```
   pip install PyQt5 Pillow numpy matplotlib scikit-image
   ```
3. Ensure you have the necessary model checkpoint files in the `checkpoints` directory:
   - `res2next-mammography-epoch=09-val_loss=0.00.ckpt`
   - `pectoral-segmentation-unet-512-epoch=06-val_dice_coeff=0.97.ckpt`
4. Update the path to the nipple detection model in the `initialize_models` method if necessary:
   ```python
   self.model_nipple_detection = nipple_detection.load_model(r'path/to/your/best.pt')
   ```

## Usage
1. Run the application:
   ```
   python mammoqc_app.py
   ```
2. Drag and drop DICOM files into the text area or use the file dialog to select files.
3. Click "Process Files" to load the DICOM images.
4. Select an image from the list to view it.
5. Click "Analyze" to perform automated analysis on the selected image.
6. View the results in the right panel and the overlaid annotations on the image.

## Code Structure
- `MainWindow`: The main application window containing all UI elements and logic.
- `ScalableImageLabel`: A custom QLabel subclass for displaying scalable images.
- Key methods:
  - `process_files`: Handles DICOM file processing.
  - `analyze_image`: Performs the automated analysis on the selected image.
  - `create_combined_image`: Creates an annotated image with analysis results.
  - `initialize_models`: Loads the machine learning models for analysis.

## Known Issues
- Large DICOM files may cause slow processing times.
- The application requires a significant amount of RAM for processing high-resolution images.

## License
[MIT Licence](../LICENSE)
