from ultralytics import YOLO
import cv2
import argparse
import os

def load_model(checkpoint_path):
    model = YOLO('yolov8m.pt')  # load an official model
    model = YOLO(r'D:\Code\MammoQC\runs\detect\train4\weights\best.pt')  # load a custom model
    return model

def draw_bbox(image_path, results):
    xyxy = results[0].boxes.xyxy[0]

    x1, y1, x2, y2 = int(xyxy[0].item()), int(xyxy[1].item()), int(xyxy[2].item()), int(xyxy[3].item())

    img = cv2.imread(image_path)

    # draw a rectangle around the detected object
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Resize the image to 800x800
    img = cv2.resize(img, (800, 800))
    return img

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
        img = draw_bbox(image_path, results)

        cv2.imshow("cropped_image", img)
        cv2.waitKey(0)
        
if __name__ == "__main__":
    main()
