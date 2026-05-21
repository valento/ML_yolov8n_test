import cv2
import os
import sys

class CameraUtils:
  def __init__(self):
    self.image_size = 640

  def capture(self, num_images=50):
    """
    Capture Images
    Args: 
      num_images: Mumber of images to capture
    """
    # images = []
    # folder to dump captures
    output_dir = "raw_captures"
    os.makedirs("raw_captures", exist_ok=True)

    cap = cv2.VideoCapture(0) # Adjust for your camera interface index: 0,1,2...
    cap_counter = 0

    while cap_counter < num_images:
      ret, frame = cap.read()
      if not ret:
        break
      
      # Show frames:
      displlay = frame.copy()
      cv2.putText(displlay, f"Captured: stick_{cap_counter}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
      cv2.putText(displlay, "SPACE = save, ESC = cancel",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
      cv2.imshow('Capture', displlay)

      key = cv2.waitKey(1) & 0xFF
      if key == 27: # ESC to stop
        break

      elif key == 32: # SPACE to save
        img_path = os.path.join(output_dir,f"stick_{cap_counter}.jpg")
        cv2.imwrite(img_path, frame)
        cap_counter += 1
        print(f"Captured image {cap_counter}/{num_images}")
      
    cap.release()
    cv2.destroyAllWindows()

   
if __name__ == "__main__":
  cam = CameraUtils()
  cam.capture(num_images=10)