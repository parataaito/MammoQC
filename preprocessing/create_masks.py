import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import os
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

def process_image(image_element, output_folder):
    image_name = image_element.get('name')
    width = int(image_element.get('width'))
    height = int(image_element.get('height'))
   
    pectoral_masks = image_element.findall('.//mask[@label="pectoral"]')
    
    if pectoral_masks:
        for mask_element in pectoral_masks:
            rle = mask_element.get('rle')
            left = int(mask_element.get('left'))
            top = int(mask_element.get('top'))
            mask_width = int(mask_element.get('width'))
            mask_height = int(mask_element.get('height'))
           
            mask = rle_decode(rle, mask_width, mask_height)
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[top:top+mask_height, left:left+mask_width] = mask
    else:
        # Create an empty mask if no pectoral tag is found
        full_mask = np.zeros((height, width), dtype=np.uint8)
   
    image = Image.fromarray(full_mask * 255)
    output_filename = os.path.splitext(image_name)[0] + '.png'
    output_path = os.path.join(output_folder, output_filename)
    image.save(output_path)

def main():
    xml_file = 'annotations.xml'
    input_folder = r'D:\Code\dcm2png\png_data'
    output_folder = r'D:\Code\dcm2png\png_masks'
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
   
    tree = ET.parse(xml_file)
    root = tree.getroot()
   
    image_elements = root.findall('.//image')
    
    for image_element in tqdm(image_elements, desc="Processing images", unit="image"):
        process_image(image_element, output_folder)

if __name__ == "__main__":
    main()