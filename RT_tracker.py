import cv2
from ultralytics import YOLO
from datetime import datetime
import os

def main():
  # Lad pt-file!
  # model = YOLO("stick_detector.pt")
  model = YOLO("v1_groups.engine") # precompiled NVIDIA-optimized-omdel to run (fused layers)

  # Galaxy Z Flip 7 camera stream
  cap = cv2.VideoCapture(0) # Adjust index to 1 or 2 if needed

  print("\nLive Tracker Active. 'q' to stop!")

  while True:
    # Clear frame buffer queue to guarantee zero live delay
    for _ in range(5):
      cap.grab()
    
    ret, frame = cap.retrieve()
    # frame_aligned = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
      break
    # frame_aligned = cv2.cvtColor(frame_aligned, cv2.COLOR_RGB2BGR)
    cv2.imshow("NOT-Rotated",frame,)

    # Raw color frame to YOLO model
    results = model(frame, stream=True, conf=0.5, imgsz=640, device=0)

    # Draw a duplicate frame to show on your screen
    display_frame = frame.copy()

    for result in results:
      boxes = result.boxes

      for box in boxes:
        # boundary pixels
        x1, y1, x2, y2 = box.xyxy.tolist()[0]
        confidence = box.conf.item()

        # --- ROBOTICS DATA ---
        # mathematically exact center of the bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Print live coordinates to the terminal for your control loops
        print(f"Target Center: ({center_x}, {center_y}) | Conf: {confidence:.2%}")

        # Draw the visual bounding box and center dot live on your desktop monitor
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(display_frame, f"Stick {confidence:.1%}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the live tracking output
    cv2.imshow("Robot High-Level CV", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
      break
    elif key == ord('p'):
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      screenshot_dir = "detections_shots"
      os.makedirs(screenshot_dir, exist_ok=True)
      snapshot_name = os.path.join(screenshot_dir, f"snap_{timestamp}.jpg")
      cv2.imwrite(snapshot_name, display_frame)

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
