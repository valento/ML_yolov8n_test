import cv2
import os

def fix_orientation():
  img_dir = "dataset/train/images"
  lbl_dir = "dataset/train/labels"

  if not os.path.exists(img_dir):
    return
  
  images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

  for img_name in images:
    # rotate Image file
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)

    # 90 ccw
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(img_path, rotated)

    # switch bounding box coords in text file
    txt_name = os.path.splitext(img_name)[0] + '.txt'
    txt_path = os.path.join(lbl_dir, txt_name)

    if os.path.exists(txt_path):
      with open(txt_path, 'r') as f:
        lines = f.readlines()

      rotated_lines = []
      for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
          continue
        cls,cx,cy,nw,nh = map(float, parts) # carefull, this takes only floats

        # Transform Rotate 90 CCW
        new_cx = cy # flip those
        new_cy = 1.0 - cx # across the Y axis
        new_nw = nh # jjust switch those
        new_nh = nw # jjust switch those

        rotated_lines.append(f"{int(cls)} {new_cx:.6f} {new_cy:.6f} {new_nw:.6f} {new_nh:.6f}\n")
      with open(txt_path, 'w') as f:
        f.writelines(rotated_lines)

    print(" Dataset Rotation complete!")

if __name__ == "__main__":
  fix_orientation()