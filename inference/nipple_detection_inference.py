from ultralytics import YOLO
import cv2
import argparse
import os
import numpy as np
import torch 

def load_model(checkpoint_path):
    model = YOLO('yolov8m.pt')  # load an official model
    model = YOLO(r'D:\Code\MammoQC\runs\detect\train4\weights\best.pt')  # load a custom model
    return model

def draw_bbox_from_path(image_path, results):
    x1, y1, x2, y2 = get_nipple_bbox(results)

    img = cv2.imread(image_path)

    # draw a rectangle around the detected object
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Resize the image to 800x800
    img = cv2.resize(img, (800, 800))
    return img

def scale_bbox(bbox, original_size, resized_size=(640, 640)):
    """
    Scale bounding box coordinates from resized image back to original image.
    
    :param bbox: List or tuple of [x1, y1, x2, y2] in resized image coordinates
    :param original_size: Tuple of (original_height, original_width)
    :param resized_size: Tuple of (resized_height, resized_width), default (640, 640)
    :return: Scaled bounding box coordinates for the original image
    """
    orig_w, orig_h = original_size
    resized_w, resized_h = resized_size
    
    # Calculate scaling factors
    x_scale = orig_w / resized_w
    y_scale = orig_h / resized_h
    
    # Unpack bbox
    x1, y1, x2, y2 = bbox
    
    # Scale coordinates
    x1_scaled = int(x1 * x_scale)
    y1_scaled = int(y1 * y_scale)
    x2_scaled = int(x2 * x_scale)
    y2_scaled = int(y2 * y_scale)
    
    return x1_scaled, y1_scaled, x2_scaled, y2_scaled

def get_nipple_bbox(results):
    xyxy = results[0].boxes.xyxy[0]

    x1, y1, x2, y2 = int(xyxy[0].item()), int(xyxy[1].item()), int(xyxy[2].item()), int(xyxy[3].item())
    return x1,y1,x2,y2

def preprocess_model_input(img):
    rgb_image = img.convert('RGB')
    img_tensor = torch.from_numpy(np.array(rgb_image.resize((640, 640))).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return img_tensor

def main():
    parser = argparse.ArgumentParser(description='Nipple Detection Inference')  
    parser.add_argument('-c', '--checkpoint_path', type=str, help='Path to the model checkpoint', default=r'D:\Code\MammoQC\runs\detect\train4\weights\best.pt')
    parser.add_argument('-i', '--image_dir', type=str, help='Path to the input image', default='inference/inference_imgs')
    args = parser.parse_args()

    # Load the model
    model = load_model(args.checkpoint_path)

    # Process each image in the directory
    for image_file in os.listdir(args.image_dir):
        # Get full path
        image_path = os.path.join(args.image_dir, image_file)   

        # Predict with the model
        results = model(image_path)  # predict on an image

        # Draw bounding box
        img = draw_bbox_from_path(image_path, results)

        cv2.imshow("cropped_image", img)
        cv2.waitKey(0)
        
if __name__ == "__main__":
    main()
