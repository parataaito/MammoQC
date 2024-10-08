import unittest
import os
import tempfile
import numpy as np
import pydicom
from PIL import Image
import xml.etree.ElementTree as ET
import sys

# Add the parent directory to the Python path to import vindr_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.vindr_preprocessing import rle_decode, dicom_to_png, create_mask, process_files

class TestVindrPreprocessing(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.test_dir, 'data')
        self.test_dicom = os.path.join(self.data_dir, 'test.dicom')
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        for root, dirs, files in os.walk(self.output_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.output_dir)

    def test_rle_decode(self):
        rle_string = "5,10,3,7"
        width, height = 5, 5
        expected_mask = np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        result = rle_decode(rle_string, width, height)
        np.testing.assert_array_equal(result, expected_mask)

    def test_dicom_to_png(self):
        png_filename, image_shape = dicom_to_png(self.test_dicom, self.output_dir)
        
        # Check if PNG file was created
        png_path = os.path.join(self.output_dir, 'images', png_filename)
        self.assertTrue(os.path.exists(png_path))
        
        # Verify PNG content
        png_image = Image.open(png_path)
        png_array = np.array(png_image)
        
        # Read original DICOM for comparison
        ds = pydicom.dcmread(self.test_dicom)
        original_array = ds.pixel_array
        
        self.assertEqual(png_array.shape, original_array.shape)
        self.assertEqual(png_array.dtype, np.uint8)
        self.assertTrue(np.all(png_array >= 0) and np.all(png_array <= 255))

    def test_create_mask_with_pectoral(self):
        xml_string = """
        <image name="test_image.png" width="100" height="100">
            <mask label="pectoral" source="manual" occluded="0" rle="20,80,20,80" left="0" top="0" width="100" height="100" z_order="0">
            </mask>
        </image>
        """
        image_element = ET.fromstring(xml_string)
        create_mask(image_element, self.output_dir, "test_image.png", (100, 100))
        
        mask_path = os.path.join(self.output_dir, 'masks', "test_image.png")
        self.assertTrue(os.path.exists(mask_path))
        
        mask = np.array(Image.open(mask_path))
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.any(mask == 255))  # Check if there's any white pixel (pectoral mask)

    def test_create_mask_without_pectoral(self):
        xml_string = """
        <image name="test_image_no_mask.png" width="100" height="100">
        </image>
        """
        image_element = ET.fromstring(xml_string)
        create_mask(image_element, self.output_dir, "test_image_no_mask.png", (100, 100))
        
        mask_path = os.path.join(self.output_dir, 'masks', "test_image_no_mask.png")
        self.assertTrue(os.path.exists(mask_path))
        
        mask = np.array(Image.open(mask_path))
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all(mask == 0))  # Check if all pixels are black (no mask)

if __name__ == '__main__':
    unittest.main()