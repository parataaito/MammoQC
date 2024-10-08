import unittest
import os
import tempfile
import numpy as np
import pydicom
from PIL import Image
import sys

# Add the parent directory to the Python path to import dcm2png
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dcm2png import dicom_to_png, process_folder

class TestDcm2Png(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, 'data')
        self.test_dicom = os.path.join(self.data_dir, 'test.dicom')
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))
        os.rmdir(self.output_dir)

    def test_dicom_to_png(self):
        # Convert DICOM to PNG
        dicom_to_png(self.test_dicom, self.output_dir)
        
        # Check if PNG file was created
        png_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        self.assertEqual(len(png_files), 1)
        
        # Verify PNG content
        png_path = os.path.join(self.output_dir, png_files[0])
        png_image = Image.open(png_path)
        png_array = np.array(png_image)
        
        # Read original DICOM for comparison
        ds = pydicom.dcmread(self.test_dicom)
        original_array = ds.pixel_array
        
        self.assertEqual(png_array.shape, original_array.shape)
        self.assertEqual(png_array.dtype, np.uint8)
        self.assertTrue(np.all(png_array >= 0) and np.all(png_array <= 255))

    def test_process_folder(self):
        # Process the folder containing the test DICOM file
        process_folder(self.data_dir, self.output_dir)
        
        # Check if PNG file was created
        png_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        self.assertEqual(len(png_files), 1)
        
        # Verify the PNG file name
        expected_name = f"test_{os.path.basename(self.data_dir)}.png"
        self.assertTrue(any(file.endswith(expected_name) for file in png_files))

if __name__ == '__main__':
    unittest.main()