# Ultralytics handles complex training loops, validation, GPU-usage
from ultralytics import YOLO

def main():
  # Ultralytics models (https://platform.ultralytics.com/ultralytics/yolov8)
  model = YOLO("yolov8n.pt") # # pretrained YOLOv8 model for robots

  # Batch infernece for now
  # buy USB-c addapter
  # results = model.predict(
  #   source = "test_images/",
  #   conf = .25,               # Confidence (only if >25% sure)
  #   save = True,              # YOLO autamitic boundary + save
  #   device = 0                # GPU
  # )

  # use a mode (train|predict|val|export|track|benchamark)
  result = model.train(
    data = "data.yaml", # Path to the dataset configuration file we made
    epochs = 50,        # Rounds to train
    imgsz = 640,
    device = 0,         # force GPU training
    workers = 4,        # CPU threads
    exist_ok = True,    # write/overwrite runs/detect/train folder
    # degrees = 180.0
  )

  print("   Training Finished!")

if __name__ == "__main__":
  main()