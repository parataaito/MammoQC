import unittest
import numpy as np
import cv2
import os
import sys

# Add the parent directory to the Python path to import vindr_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.pnd import pectoral_nipple_distance

class TestPectoralNippleDistance(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join('tests', 'data', 'pnd')

    def test_right_mlo(self):
        img_path = os.path.join(self.data_dir, 'img_Right_MLO.png')
        mask_path = os.path.join(self.data_dir, 'msk_Right_MLO.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img_color, intersection_point, distance = pectoral_nipple_distance(
            img, mask, 1562.39, 2345.40, 1756.72, 2559.83, "Right", "MLO"
        )
        
        self.assertIsNotNone(intersection_point)
        self.assertGreater(distance, 0)
        self.assertAlmostEqual(intersection_point, (2791, 2079))
        self.assertEqual(img_color.shape[:2], img.shape)
        self.assertEqual(img_color.shape[2], 3)  # Check if output is color image

    def test_left_mlo(self):
        img_path = os.path.join(self.data_dir, 'img_Left_MLO.png')
        mask_path = os.path.join(self.data_dir, 'msk_Left_MLO.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img_color, intersection_point, distance = pectoral_nipple_distance(
            img, mask, 839.46, 1840.58, 1008.32, 2057.69, "Left", "MLO"
        )
        
        self.assertIsNotNone(intersection_point)
        self.assertAlmostEqual(intersection_point, (78, 1613))
        self.assertGreater(distance, 0)
        self.assertEqual(img_color.shape[:2], img.shape)
        self.assertEqual(img_color.shape[2], 3)

    def test_left_cc(self):
        img_path = os.path.join(self.data_dir, 'img_Left_CC.png')
        mask_path = os.path.join(self.data_dir, 'msk_Left_CC.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img_color, intersection_point, distance = pectoral_nipple_distance(
            img, mask, 677.86, 1390.51, 805.18, 1544.63, "Left", "CC"
        )
        
        self.assertIsNotNone(intersection_point)
        self.assertAlmostEqual(intersection_point, (0, 1467))
        self.assertGreater(distance, 0)
        self.assertEqual(img_color.shape[:2], img.shape)
        self.assertEqual(img_color.shape[2], 3)

    def test_right_cc(self):
        img_path = os.path.join(self.data_dir, 'img_Right_CC.png')
        mask_path = os.path.join(self.data_dir, 'msk_Right_CC.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        img_color, intersection_point, distance = pectoral_nipple_distance(
            img, mask, 1729.91, 1393.86, 1837.13, 1568.09, "Right", "CC"
        )
        
        self.assertIsNotNone(intersection_point)
        self.assertAlmostEqual(intersection_point, (2799, 1480))
        self.assertGreater(distance, 0)
        self.assertEqual(img_color.shape[:2], img.shape)
        self.assertEqual(img_color.shape[2], 3)

    def test_invalid_image_path(self):
        with self.assertRaises(cv2.error):
            img_path = os.path.join(self.data_dir, 'non_existent_image.png')
            mask_path = os.path.join(self.data_dir, 'msk_Right_CC.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pectoral_nipple_distance(img, mask, 0, 0, 10, 10, "Right", "CC")

    def test_invalid_orientation(self):
        with self.assertRaises(AssertionError):
            img_path = os.path.join(self.data_dir, 'img_Right_CC.png')
            mask_path = os.path.join(self.data_dir, 'msk_Right_CC.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pectoral_nipple_distance(img, mask, 0, 0, 10, 10, "Invalid", "CC")

    def test_invalid_view(self):
        with self.assertRaises(AssertionError):
            img_path = os.path.join(self.data_dir, 'img_Right_CC.png')
            mask_path = os.path.join(self.data_dir, 'msk_Right_CC.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pectoral_nipple_distance(img, mask, 0, 0, 10, 10, "Right", "Invalid")

if __name__ == '__main__':
    unittest.main()