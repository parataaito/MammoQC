<p align="center"><img align="center" width="280" src="docs/icon.png"/></p>
<h3 align="center">MammoQC, an open-source Patient Positioning Quality Control tool for X-Ray mammography</h3>
<hr>

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,pytorch,opencv,qt" />
  </a>
</p>
<hr>
MammoQC is a comprehensive tool for mammography quality control and analysis. It includes three main components: nipple detection, pectoral muscle segmentation, and view-orientation classification. This project uses advanced deep learning techniques to analyze mammogram images and provide valuable insights for medical professionals.

<img src="docs/images/mammoqc_exemple1.png" width="500" height="300">
<img src="docs/images/mammoqc_exemple2.png" width="500" height="300">

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Training](#training)
- [Inference](#inference)
- [Image Processing](#image_processing)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Future Work Checklist](#future-work-checklist)
- [License](#license)

## Features

1. **Nipple Detection**: Utilizes YOLOv8 to accurately locate nipples in mammogram images.
2. **Pectoral Muscle Segmentation**: Implements a U-Net architecture to segment pectoral muscles in mammograms.
3. **View-Orientation Classification**: Uses a Res2Next model to classify mammogram views (CC/MLO) and orientations (Left/Right).
4. **Pectoral Nipple Distance**: Combine multiple models and image processing techniques to determine the PND

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/MammoQC.git
   cd MammoQC
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install PyTorch:
   **Important**: Install PyTorch separately using the command from the official PyTorch website. Select the relevant version of CUDA for your system. For example:
   ```
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset Information

[Dataset Documentation](preprocessing/README.md)

## Training

[Training Documentation](train/README.md)

## Inference

[Inference Documentation](inference/README.md)

## Inference

[Image Processing Documentation](processing/README.md)

## User Interface

[MammoQC App](ui/README.md)

## Dependencies

MammoQC relies on several Python libraries. The main dependencies include:

- PyTorch and torchvision (install separately as mentioned in the Installation section)
- PyTorch Lightning
- Ultralytics (for YOLOv8)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- timm (for Res2Next model)

For a complete list of dependencies and their versions, please refer to the `requirements.txt` file in the repository.

**Note**: Some dependencies may require specific versions to ensure compatibility. It's recommended to use a virtual environment and install dependencies exactly as specified in the `requirements.txt` file.

## Future Work Checklist

Here's a list of features and improvements planned for the MammoQC project:

### Data Preprocessing
- [x] Package the preprocessing pipeline for the VINDR dataset
  - [x] Create a script to download and extract the VINDR dataset
  - [x] Implement automatic data preprocessing steps (DICOM to PNG conversion, masks creation, etc.)
  - [x] Document the preprocessing steps in detail

### Algorithm Development
- [ ] Create a full algorithm for Patient Position Quality Control
  - [x] Merge all existing models (view classification, orientation detection, etc.)
  - [ ] Implement additional processing steps:
    - [ ] Breast boundary detection
    - [ ] Nipple visibility detection
  - [ ] Develop a scoring system for overall positioning quality
  - [ ] Create a comprehensive report generation feature

### User Interface
- [ ] Add a user-friendly UI
  - [x] Design a clean and intuitive interface
  - [x] Implement file upload functionality for mammogram images
  - [x] Create a results display page with visualizations
  - [ ] Add batch processing capabilities
  - [ ] Implement result saving features

### Deployment
- [ ] Dockerize the application

### Testing and Validation
- [ ] Develop a comprehensive test suite
  - [ ] Unit tests for individual components
  - [ ] Integration tests for the full algorithm
  - [ ] Performance benchmarking on various hardware configurations

### Future Enhancements
- [ ] Handle other modalities (MRI?)
- [ ] Develop additional QC modules (e.g., compression, exposure)

## License

This project is licensed under the MIT Licence. See the `LICENSE` file for details.

















