import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from train.view_orientation_classifier import Res2NextLightningModule

def load_model(checkpoint_path):
    # Load the model to CPU first
    model = Res2NextLightningModule.load_from_checkpoint(checkpoint_path, map_location=torch.device('cuda'))
    model.eval()
    return model

def preprocess_image_from_path(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def preprocess_image_from_numpy(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    # Ensure the model and input tensor are on the same device
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        view_output, orientation_output = model(image_tensor)
        view_prob = torch.softmax(view_output, dim=1)
        orientation_prob = torch.softmax(orientation_output, dim=1)
        
        view_pred = torch.argmax(view_prob, dim=1).item()
        orientation_pred = torch.argmax(orientation_prob, dim=1).item()
        
        view_label = 'CC' if view_pred == 0 else 'MLO'
        orientation_label = 'Right' if orientation_pred == 0 else 'Left'
        
        return view_label, orientation_label, view_prob.squeeze().tolist(), orientation_prob.squeeze().tolist()

def main():
    parser = argparse.ArgumentParser(description='Mammography View and Orientation Inference')
    parser.add_argument('-c', '--checkpoint_path', type=str, help='Path to the model checkpoint', default='checkpoints/res2next-mammography-epoch=09-val_loss=0.00.ckpt')
    parser.add_argument('-i', '--image_dir', type=str, help='Path to the input image', default='inference/inference_imgs')
    args = parser.parse_args()

    try:
        # Load the model
        model = load_model(args.checkpoint_path)
        
        # Process each image in the directory
        for image_file in os.listdir(args.image_dir):
            # Get full path
            image_path = os.path.join(args.image_dir, image_file)
            
            # Preprocess the image
            image_tensor = preprocess_image_from_path(image_path)
            
            # Make prediction
            view, orientation, view_probs, orientation_probs = predict(model, image_tensor)
            
            # Print results
            print(f"Predicted for {image_file}: {orientation} {view}")
            print(f"Orientation Probabilities: Right: {orientation_probs[0]:.4f}, Left: {orientation_probs[1]:.4f}")
            print(f"View Probabilities: CC: {view_probs[0]:.4f}, MLO: {view_probs[1]:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()