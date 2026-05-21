from ultralytics import YOLO
# import cv2
import os

def main():
    # Load the PyTorch weights and model
    model = YOLO("best.pt")

    # Path to image for test
    test_image_path = "dataset/train/images/stick_6.jpg" 
    
    if not os.path.exists(test_image_path):
      print(f"Could not find test image at {test_image_path}. Check your filename!")
      return

    print(f"Analyzing {test_image_path} with your custom AI...")

    # Run
    results = model.predict(source=test_image_path, conf=0.5, save=True, device=0)

    # Robot Data to publish (if ROS node)
    for result in results:
      boxes = result.boxes
      for box in boxes:
        # bounding boxcoords
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # confidence score (0.0 to 1.0)
        confidence = box.conf[0].item()
        
        print("\nROBOT DATA OUTPUT:")
        print(f"  Stick!: {confidence:.2%}")
        # print(f"  Bounding Box: Top-Left=({int(x1)}, {int(y1)}), Bottom-Right=({int(x2)}, {int(y2)})")
        
        # Math the center pixel
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        print(f"  Target Center gripper0: X={center_x}, Y={center_y}")

if __name__ == '__main__':
    main()
