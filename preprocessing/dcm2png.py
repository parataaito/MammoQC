import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image

def dicom_to_png(dicom_file, output_folder):
    # Read the DICOM file
    ds = pydicom.dcmread(dicom_file)
    
    # Convert to float to avoid overflow or underflow losses
    image = apply_voi_lut(ds.pixel_array, ds).astype(float)
    
    # Rescale to 0-255 and convert to uint8
    image = ((image - image.min()) / (image.max() - image.min()) * 255.0).astype(np.uint8)
    
    # Create a PIL Image
    final_image = Image.fromarray(image)
    
    # Generate a unique filename for the PNG
    base_filename = os.path.splitext(os.path.basename(dicom_file))[0]
    png_filename = f"{base_filename}_{os.path.basename(os.path.dirname(dicom_file))}.png"
    png_path = os.path.join(output_folder, png_filename)
    
    # Ensure the filename is unique
    counter = 1
    while os.path.exists(png_path):
        png_filename = f"{base_filename}_{os.path.basename(os.path.dirname(dicom_file))}_{counter}.png"
        png_path = os.path.join(output_folder, png_filename)
        counter += 1
    
    # Save as PNG
    final_image.save(png_path)
    print(f"Converted {dicom_file} to {png_path}")

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Walk through all subfolders recursively
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.dicom', '.dcm')):
                dicom_file = os.path.join(root, file)
                dicom_to_png(dicom_file, output_folder)

if __name__ == "__main__":
    input_folder = r'D:\Data\physionet.org\files\vindr-mammo\1.0.0\images'
    output_folder = 'png_data'
    process_folder(input_folder, output_folder)