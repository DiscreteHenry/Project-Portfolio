from ultralytics import YOLO
import os

# --- Configuration ---
DATA_YAML_PATH = 'dataset.yaml'
MODEL_VARIANT = 'yolov8n.pt'  # Choose your YOLOv8 model: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
EPOCHS = 20                 # Number of training epochs
IMG_SIZE = 1024              # Image size for training (larger for small objects)
BATCH_SIZE = 2               # Adjust based on your GPU memory
DEVICE = 'cpu'                   # GPU device ID (0 for first GPU, or 'cpu' for CPU training)
PROJECT_NAME = 'crystal_counting_yolov8'
EXPERIMENT_NAME = 'initial_image_weak_model'      # Experiment name for runs directory
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detection during validation
IOU_THRESHOLD = 0.7          # NMS IoU threshold during validation

# --- Main Training Function ---
def train_yolov8():
    print(f"Starting YOLOv8 training with {MODEL_VARIANT}...")

    # Load a pre-trained YOLOv8 model
    # 'n' is nano (fastest, smallest), 's' is small, 'm' is medium, etc.
    model = YOLO(MODEL_VARIANT)

    # Train the model
    # The 'data' argument points to your dataset.yaml
    # 'imgsz' is crucial for small objects like crystals.
    # 'project' and 'name' organize output in the 'runs/detect' directory.
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        conf=CONFIDENCE_THRESHOLD, # Confidence threshold for validation metrics
        iou=IOU_THRESHOLD,         # IoU threshold for NMS in validation metrics
        # Additional hyperparameters can be added here, e.g., learning rate, weight decay, etc.
        # Check YOLOv8 docs for full list: https://docs.ultralytics.com/usage/cfg/#train
        patience=20, # Stop if no improvement in 20 epochs
        # augment=True # Default True for most transforms, can be customized
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {model.trainer.save_dir}/weights/best.pt")
    print(f"Last model saved to: {model.trainer.save_dir}/weights/last.pt")

if __name__ == '__main__':
    # Ensure you are in the yolov8_training directory when running this script
    # or adjust paths accordingly.
    # e.g., cd yolov8_training
    # python train_model.py
    train_yolov8()