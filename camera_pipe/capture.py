import cv2
import os
import argparse
import numpy as np

class ImageCapture:
  def __init__(self, min_gray=0, max_gray=130, min_area=15090.5, max_area=50090.5):
    self.min_gray = min_gray
    self.max_gray = max_gray
    self.min_area = min_area
    self.max_area = max_area

  def capture(self, num_images=50, start_idx=0):
    """
    Capture Images
    Args: 
      num_images: Mumber of images to capture
    """
    # images = []
    # folder to dump captures
    output_dir = "raw_captures"
    img_dir = "dataset/train/images"
    lbl_dir = "dataset/train/labels"
    os.makedirs("raw_captures", exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    camera = cv2.VideoCapture(0) # Adjust for your camera interface index: 0,1,2...
    cap_counter = start_idx

    while cap_counter < num_images+start_idx:
      # warming up
      for _ in range(15):
        camera.read()

      ret, frame = camera.read()
      if not ret:
        break

      h, w, _ = frame.shape

      # mask
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      mask = cv2.inRange(gray, self.min_gray, self.max_gray)
      mask = cv2.bitwise_not(mask)

      # close lose ages
      kernel = np.ones((3,3), np.uint8)
      binary_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
      cv2.rectangle(binary_clean, (0, 0), (w, h), 0, thickness=10)

      # boundaries
      contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
      # Show frames:
      displlay = frame.copy()
      current_frame_yolo_lines = []

      for contour in contours:
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
          continue
        print(f"Area: {area}")
        # draw a bounding box
        x,y,box_w,box_h = cv2.boundingRect(contour)
        cv2.rectangle(displlay, (x,y), (x+box_w, y+box_h), (200,255,0),2)

        # Calculate normalized YOLO positions (0.0 to 1.0)
        cx = (x + (box_w / 2.0)) / w
        cy = (y + (box_h / 2.0)) / h
        nw = box_w / w
        nh = box_h / h

        # cache yolo string lines
        current_frame_yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

      cv2.putText(displlay, f"Captured: stick_{cap_counter}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
      cv2.putText(displlay, "SPACE = save, ESC = cancel",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
      cv2.imshow("Mask", binary_clean)
      cv2.imshow("Capture", displlay)

      key = cv2.waitKey(1) & 0xFF
      if key == 27: # ESC to stop
        break

      elif key == 32: # SPACE to save
        if not current_frame_yolo_lines:
          print("No distinct outlines... CLick blpcked")
          continue

        # save frame
        unique_name = "stick"
        img_name = f"{unique_name}_{cap_counter}.jpg"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, frame)
        
        # save label
        lable_name = f"{unique_name}_{cap_counter}.txt"
        label_path = os.path.join(lbl_dir, lable_name)
        with open(label_path, 'w') as f:
          f.writelines(current_frame_yolo_lines)
        
        cap_counter += 1
        
        print(f"Saved image/label pair: {img_name}/{lable_name}")
      
    camera.release()
    cv2.destroyAllWindows()

   
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Advanced YOLO bounderies Live capture")

  parser.add_argument(
    "--num_images",
    type=int,
    default=20,
    help="Number of images to capture"
  )

  parser.add_argument(
    "--max_gray",
    type=int,
    default=145,
    help="Maximum mask threshold"
  )

  parser.add_argument(
    "--min_area",
    type=int,
    default=15000,
    help="Minimal Area to capture"
  )
  parser.add_argument(
    "--max_area",
    type=int,
    default=25000,
    help="Maxumum Area to capture"
  )
  parser.add_argument(
    "--start_index",
    type=int,
    default=0,
    help="Save name from (Default 0)"
  )
  args = parser.parse_args()
  cam = ImageCapture(min_gray=0, max_gray=args.max_gray, min_area=args.min_area, max_area=args.max_area)
  cam.capture(num_images=args.num_images, start_idx=args.start_index)