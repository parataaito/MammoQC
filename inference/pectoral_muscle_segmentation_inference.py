import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys 
import argparse
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from train.pectoral_muscle_segmentator_512 import PectoralSegmentation

def load_model(checkpoint_path):
    # Determine if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = PectoralSegmentation.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path, device, target_size=(512, 512)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def predict(model, image):
    with torch.no_grad():
        output = model(image)
    return torch.sigmoid(output)

def visualize_result(original_image, mask):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(original_image, cmap='gray')
    ax2.imshow(mask, alpha=0.5, cmap='jet')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def main():
    parser = argparse.ArgumentParser(description='Pectoral Muscle Segmentation Inference')  
    parser.add_argument('-c', '--checkpoint_path', type=str, help='Path to the model checkpoint', default='checkpoints/pectoral-segmentation-unet-512-epoch=06-val_dice_coeff=0.97.ckpt')
    parser.add_argument('-i', '--image_dir', type=str, help='Path to the input image', default='inference/inference_imgs')
    args = parser.parse_args()
    
    # Specify the path to your checkpoint file
    checkpoint_path = args.checkpoint_path
    
    # Load the trained model
    model, device = load_model(checkpoint_path)
    
    # Specify the directory containing the images for inference
    image_dir = args.image_dir
    
    # Process each image in the directory
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            
            # Preprocess the image
            processed_image = preprocess_image(image_path, device)
            
            # Make a prediction
            prediction = predict(model, processed_image)
            
            # Convert prediction to numpy array and resize to original image size
            mask = prediction.squeeze().cpu().numpy()
            original_image = Image.open(image_path).convert('L')
            mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_image.size)
            
            # Visualize the result
            visualize_result(np.array(original_image), np.array(mask))

if __name__ == "__main__":
    main()