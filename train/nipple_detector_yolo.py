from ultralytics import YOLO

if __name__ == "__main__":
# Load a larger model
    # model = YOLO('yolov8m.pt')  # medium-sized model for better accuracy
    model = YOLO('train/yolo11m.pt')  # medium-sized model for better accuracy

    # Train the model with adjusted parameters
    # results = model.train(
    #     data="D:/Data/datasets/vindr_yolo/data.yaml",
    #     epochs=200,  # Increase number of epochs
    #     imgsz=640,   # Increase image size
    #     batch=16,    # Adjust batch size based on your GPU memory
    #     lr0=1e-6,    # Lower initial learning rate
    #     lrf=1e-4,    # Final learning rate
    #     warmup_epochs=5,  # Warmup epochs
    #     augment=True,  # Enable built-in augmentations
    #     workers=8,
    #     patience=50,  # Early stopping patience
    # )
    model.train(data="D:/Data/datasets/vindr_yolo/data.yaml", imgsz=640, epochs=200, batch=32, lr0=1e-5, workers=8)

    # Validate the model
    metrics = model.val()

    # Export the model
    success = model.export(format="onnx")

    print(f"Validation metrics: {metrics}")
    print(f"Model export successful: {success}")