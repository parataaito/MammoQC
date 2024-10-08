from ultralytics import YOLO
import argparse

def main(args):
    # Load the model
    model = YOLO(args.model)

    # Train the model
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        workers=args.workers
    )

    # Validate the model
    metrics = model.val()

    # Export the model
    success = model.export(format=args.export_format)

    print(f"Validation metrics: {metrics}")
    print(f"Model export successful: {success}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate a YOLO model for nipple detection")
    parser.add_argument("--model", type=str, default="train/yolo11m.pt", help="Path to the YOLO model")
    parser.add_argument("--data", type=str, default="D:/Data/datasets/vindr_yolo/data.yaml", help="Path to the data configuration file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr0", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for data loading")
    parser.add_argument("--export_format", type=str, default="onnx", help="Format to export the model (e.g., 'onnx', 'torchscript')")
    
    args = parser.parse_args()
    main(args)