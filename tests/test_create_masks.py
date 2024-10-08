import unittest
import numpy as np
from PIL import Image
import os
import tempfile
import xml.etree.ElementTree as ET
import sys

# Add the parent directory to the Python path to import create_masks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.create_masks import rle_decode, process_image

class TestCreateMasks(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

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

    def test_process_image_with_pectoral_mask(self):
        xml_string = """
        <image id="1" name="test_image.png" width="2800" height="3518">
            <mask label="pectoral" source="manual" occluded="0" rle="20, 554, 20, 554, 19, 555" left="2226" top="0" width="574" height="1813" z_order="0">
            </mask>
            <box label="nipple" source="manual" occluded="0" xtl="1562.39" ytl="2345.40" xbr="1756.72" ybr="2559.83" z_order="0">
            </box>
            <tag label="Right" source="manual">
            </tag>
            <tag label="nipple_in" source="manual">
            </tag>
            <tag label="MLO" source="manual">
            </tag>
        </image>
        """
        root = ET.fromstring(xml_string)
        
        process_image(root, self.temp_dir)
        
        output_file = os.path.join(self.temp_dir, "test_image.png")
        self.assertTrue(os.path.exists(output_file))
        
        mask = np.array(Image.open(output_file))
        self.assertEqual(mask.shape, (3518, 2800))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.any(mask == 255))  # Check if there's any white pixel (pectoral mask)

    def test_process_image_without_pectoral_mask(self):
        xml_string = """
        <image id="0" name="test_image_no_mask.png" width="2800" height="3518">
            <box label="nipple" source="manual" occluded="0" xtl="2068.31" ytl="1400.56" xbr="2296.14" ybr="1651.85" z_order="0">
            </box>
            <tag label="Right" source="manual">
            </tag>
            <tag label="CC" source="manual">
            </tag>
            <tag label="nipple_out" source="manual">
            </tag>
        </image>
        """
        root = ET.fromstring(xml_string)
        
        process_image(root, self.temp_dir)
        
        output_file = os.path.join(self.temp_dir, "test_image_no_mask.png")
        self.assertTrue(os.path.exists(output_file))
        
        mask = np.array(Image.open(output_file))
        self.assertEqual(mask.shape, (3518, 2800))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all(mask == 0))  # Check if all pixels are black (no mask)

    def test_process_image_invalid_structure(self):
        xml_string = """
        <image id="2" name="invalid_image.png">
            <invalid_tag></invalid_tag>
        </image>
        """
        root = ET.fromstring(xml_string)
        
        with self.assertRaises(TypeError):
            process_image(root, self.temp_dir)

if __name__ == '__main__':
    unittest.main()