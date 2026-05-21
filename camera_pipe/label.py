import cv2
import os
import numpy as np

def run_batch_labeling():
    # paths
    input_dir = "raw_captures"
    labels_dir = "dataset/train/labels"
    images_dir = "dataset/train/images"
    
    # Create the output target folders
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    if not os.path.exists(input_dir) or not os.listdir(input_dir):
      print(f"Error: No images found in '{input_dir}'")
      return

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing {len(img_files)} images using your custom threshold profile...")

    for img_name in img_files:
      # Load raw color image
      color_path = os.path.join(input_dir, img_name)
      img = cv2.imread(color_path)
      h, w, _ = img.shape

      # qpply my Masking Limits
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # mask
      mask = cv2.inRange(gray, 0, 156) 
      
      # noises
      kernel = np.ones((5,5), np.uint8)
      cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

      # fnd boundaries
      contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      yolo_lines = []
      for cnt in contours:
        # Filter out tiny pixel dust
        if cv2.contourArea(cnt) < 500: 
          continue

        # Get the pixel coordinates of the bounding box
        xmin, ymin, box_w, box_h = cv2.boundingRect(cnt)

        # Normalize to YOLO Format(0 to 1.0)
        center_x = (xmin + (box_w / 2.0)) / w
        center_y = (ymin + (box_h / 2.0)) / h
        norm_w = box_w / w
        norm_h = box_h / h

        # Class 0 = 'pallet_detail'
        yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

      if yolo_lines:
        # text file label
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        with open(os.path.join(labels_dir, txt_name), 'w') as f:
          f.writelines(yolo_lines)
        
        # Save pristine raw COLOR image to the training folder
        cv2.imwrite(os.path.join(images_dir, img_name), img)
        print(f" Generated label and copied image for: {img_name}")
      else:
        print(f"Skipped {img_name}: No mask.")

    print("\n Images + Labels synced.")

if __name__ == "__main__":
    run_batch_labeling()
