import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import numpy as np

def dicom_to_image(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    pixel_spacing = [float(val) for val in ds.ImagerPixelSpacing]
    image = apply_voi_lut(ds.pixel_array, ds).astype(float)
    image = ((image - image.min()) / (image.max() - image.min()) * 255.0).astype(np.uint8)
    final_image = Image.fromarray(image)
    return final_image, pixel_spacing