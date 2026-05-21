#!/usr/bin/env python3
import cv2
import numpy as np
import sys

class MaskingUtils:
    """Utilities for creating masks based on pixel intensity ranges"""
    
    @staticmethod
    def create_intensity_mask(image, lower, upper, invert=False):
        # Create mask: white where pixels are in range
        mask = cv2.inRange(image, lower, upper)
        
        # Invert if needed (black out the range instead of keeping it)
        if invert:
            mask = cv2.bitwise_not(mask)
        return mask
    
    @staticmethod
    def apply_mask(image, mask):
        return cv2.bitwise_and(image, image, mask=mask)


def nothing(x):
    pass


def interactive_masking():
    # Change this index (0, 1, 2) until your Galaxy Z Flip 7 stream appears
    camera_index = 0
    print(f"Opening camera interface index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera index {camera_index}")
        return 1

    # Create window with trackbars
    cv2.namedWindow('Masking Tool')
    
    # Trackbars matching your original layout
    cv2.createTrackbar('Floor Lower', 'Masking Tool', 0, 255, nothing)
    cv2.createTrackbar('Floor Upper', 'Masking Tool', 60, 255, nothing)
    cv2.createTrackbar('Pallet Lower', 'Masking Tool', 40, 255, nothing)
    cv2.createTrackbar('Pallet Upper', 'Masking Tool', 100, 255, nothing)
    cv2.createTrackbar('Mode', 'Masking Tool', 0, 2, nothing)  # 0=both, 1=floor only, 2=pallet only
    
    print("\n=== Interactive Masking Tool ===")
    print("Move your sticks under the live camera feed.")
    print("Adjust trackbars to find optimal background cleaning thresholds.")
    print("Press 'q' to quit.")
    print("Press 's' to print current parameters to terminal.")
    print("=================================")
    
    while True:
        # LIVE CAPTURE: Read current frame dynamically inside the loop
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab live frame.")
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get active trackbar values
        floor_lower = cv2.getTrackbarPos('Floor Lower', 'Masking Tool')
        floor_upper = cv2.getTrackbarPos('Floor Upper', 'Masking Tool')
        pallet_lower = cv2.getTrackbarPos('Pallet Lower', 'Masking Tool')
        pallet_upper = cv2.getTrackbarPos('Pallet Upper', 'Masking Tool')
        mode = cv2.getTrackbarPos('Mode', 'Masking Tool')
        
        # Create masks based on your original configuration settings
        if mode == 0:  # Both
            mask_floor = MaskingUtils.create_intensity_mask(gray, floor_lower, floor_upper, invert=True)
            mask_pallet = MaskingUtils.create_intensity_mask(gray, pallet_lower, pallet_upper, invert=True)
            combined_mask = cv2.bitwise_and(mask_floor, mask_pallet)
            result = MaskingUtils.apply_mask(gray, combined_mask)
            title = f"Floor:[{floor_lower}-{floor_upper}] + Pallet:[{pallet_lower}-{pallet_upper}]"
        
        elif mode == 1:  # Floor only
            mask_floor = MaskingUtils.create_intensity_mask(gray, floor_lower, floor_upper, invert=True)
            result = MaskingUtils.apply_mask(gray, mask_floor)
            title = f"Floor Only: [{floor_lower}-{floor_upper}]"
        
        else:  # Pallet only
            mask_pallet = MaskingUtils.create_intensity_mask(gray, pallet_lower, pallet_upper, invert=True)
            result = MaskingUtils.apply_mask(gray, mask_pallet)
            title = f"Pallet Only: [{pallet_lower}-{pallet_upper}]"
        
        # Display the output view
        cv2.imshow('Masking Tool', result)
        
        # Listen for key interrupts natively
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n=== Calibration Thresholds Found! ===")
            print(f"Floor Limits:  lower={floor_lower}, upper={floor_upper}")
            print(f"Pallet Limits: lower={pallet_lower}, upper={pallet_upper}")
            print("=======================================")
            
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(interactive_masking())
