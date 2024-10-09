import os
import argparse
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm

def rle_decode(rle_string, width, height):
    counts = [int(x) for x in rle_string.split(',')]
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y = 0, 0
    for i, count in enumerate(counts):
        value = i % 2
        for _ in range(count):
            if y >= height:
                break
            mask[y, x] = value
            x += 1
            if x == width:
                x = 0
                y += 1
    return mask

def dicom_to_png(dicom_file, output_folder):
    ds = pydicom.dcmread(dicom_file)
    image = apply_voi_lut(ds.pixel_array, ds).astype(float)
    image = ((image - image.min()) / (image.max() - image.min()) * 255.0).astype(np.uint8)
    final_image = Image.fromarray(image)
    
    base_filename = os.path.splitext(os.path.basename(dicom_file))[0]
    png_filename = f"{base_filename}_{os.path.basename(os.path.dirname(dicom_file))}.png"
    png_path = os.path.join(output_folder, 'images', png_filename)
    
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    final_image.save(png_path)
    return png_filename, image.shape

def create_mask(image_element, output_folder, filename, image_shape):
    width = int(image_element.get('width'))
    height = int(image_element.get('height'))
    
    full_mask = np.zeros(image_shape, dtype=np.uint8)
    
    pectoral_masks = image_element.findall('.//mask[@label="pectoral"]')
    
    if pectoral_masks:
        for mask_element in pectoral_masks:
            rle = mask_element.get('rle')
            left = int(mask_element.get('left'))
            top = int(mask_element.get('top'))
            mask_width = int(mask_element.get('width'))
            mask_height = int(mask_element.get('height'))
            
            mask = rle_decode(rle, mask_width, mask_height)
            full_mask[top:top+mask_height, left:left+mask_width] = mask
    
    image = Image.fromarray(full_mask * 255)
    output_path = os.path.join(output_folder, 'masks', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

def process_files(input_folder, output_folder, annotations_file):
    tree = ET.parse(annotations_file)
    root = tree.getroot()
    image_elements = {elem.get('name'): elem for elem in root.findall('.//image')}
    
    all_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.dicom', '.dcm')):
                all_files.append(os.path.join(root, file))
    
    for dicom_file in tqdm(all_files, desc="Processing files", unit="file"):
        png_filename, image_shape = dicom_to_png(dicom_file, output_folder)
        
        if png_filename in image_elements:
            create_mask(image_elements[png_filename], output_folder, png_filename, image_shape)
        else:
            # Create an empty mask with the same size as the image
            empty_mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
            empty_mask_path = os.path.join(output_folder, 'masks', png_filename)
            os.makedirs(os.path.dirname(empty_mask_path), exist_ok=True)
            empty_mask.save(empty_mask_path)

def main():
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG and create binary masks.")
    parser.add_argument("-i", "--input_folder", help="Path to the VINDR folder containing DICOM files, such as 'vindr_folder/1.0.0/images'", default=r"D:\Data\physionet.org\files\vindr-mammo\1.0.0\images")
    parser.add_argument("-o", "--output_folder", help="Path to your working directory for PNG images and binary masks", default=r"D:\test")
    parser.add_argument("-a", "--annotations_file", help="Path to the CVAT XML annotations file", default=r".\annotations.xml")
    
    args = parser.parse_args()
    
    process_files(args.input_folder, args.output_folder, args.annotations_file)

if __name__ == "__main__":
    main()