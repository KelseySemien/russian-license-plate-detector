import cv2
import numpy as np
import pytesseract
import os

class RussianLicensePlateRecognizer:
    def __init__(self):
        
        cascade_paths = [
            "haarcascade_russian_plate_number.xml",
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml",
            os.path.expanduser("~/Desktop/plate_detection/haarcascade_russian_plate_number.xml")
        ]
        
        self.plate_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                self.plate_cascade = cv2.CascadeClassifier(path)
                if not self.plate_cascade.empty():
                    print(f"✓ Loaded cascade from: {path}")
                    break
        
        if self.plate_cascade is None or self.plate_cascade.empty():
            print("✗ ERROR: Could not load cascade classifier!")
            print("\nPlease download it manually:")
            print("cd ~/Desktop/plate_detection")
            print("curl -o haarcascade_russian_plate_number.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml")
            print("\nOr copy from OpenCV:")
            print(f"cp {cv2.data.haarcascades}haarcascade_russian_plate_number.xml ~/Desktop/plate_detection/")
    
    def preprocess_for_detection(self, image):
        """Preprocess image to improve plate detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
       
        strategies = []
        
        # Strategy 1: Original
        strategies.append(gray)
        
        # Strategy 2: CLAHE (contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        strategies.append(clahe.apply(gray))
        
        # Strategy 3: Equalize histogram
        strategies.append(cv2.equalizeHist(gray))
        
        # Strategy 4: Adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        strategies.append(binary)
        
        return strategies
    
    def detect_license_plate(self, image):
        """Try multiple strategies to detect license plate"""
        if self.plate_cascade is None or self.plate_cascade.empty():
            return None
        
        # Get multiple preprocessing versions
        preprocessed_images = self.preprocess_for_detection(image)
        
        # Parameter sets to try
        param_sets = [
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (80, 30), 'maxSize': (400, 150)},
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (60, 25), 'maxSize': (500, 200)},
            {'scaleFactor': 1.2, 'minNeighbors': 4, 'minSize': (70, 28), 'maxSize': (450, 180)},
        ]
        
        best_plate = None
        best_area = 0
        
        for i, prep_img in enumerate(preprocessed_images):
            for params in param_sets:
                plates = self.plate_cascade.detectMultiScale(prep_img, **params)
                
                if len(plates) > 0:
                    # Take largest plate
                    for (x, y, w, h) in plates:
                        area = w * h
                        if area > best_area:
                            best_area = area
                            best_plate = (x, y, w, h)
                    
                    if best_plate is not None:
                        print(f"  ✓ Detected with strategy {i+1}, params: scale={params['scaleFactor']}")
                        return best_plate
        
        return None
    
    def process_image(self, image_path):
        """Process single image with debugging"""
        print(f"\n--- Processing {os.path.basename(image_path)} ---")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ✗ Could not read image")
            return None, None
        
        print(f"  Image size: {img.shape}")
        
        #  detection
        plate_rect = self.detect_license_plate(img)
        
        if plate_rect is None:
            print(f"  ✗ No plate detected after trying all strategies")
            # Save preprocessed versions for debugging
            preprocessed = self.preprocess_for_detection(img)
            for i, prep in enumerate(preprocessed):
                cv2.imwrite(f"debug_prep_{i}_{os.path.basename(image_path)}", prep)
            print(f"  Debug images saved for manual inspection")
            return img, None
        
        x, y, w, h = plate_rect
        print(f"  ✓ Plate detected at: x={x}, y={y}, w={w}, h={h}")
        
        # Draw bounding box
        result_img = img.copy()
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(result_img, "License Plate", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Extract plate region
        plate_region = img[y:y+h, x:x+w]
        cv2.imwrite(f"extracted_plate_{os.path.basename(image_path)}", plate_region)
        
        return result_img, plate_region


def main():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    project_dir = os.path.join(desktop, "plate_detection")
    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)
    
    print("\n" + "="*60)
    print("RUSSIAN LICENSE PLATE DETECTION - DEBUG VERSION")
    print("="*60)
    
    # Check if cascade exists
    cascade_file = "haarcascade_russian_plate_number.xml"
    if not os.path.exists(cascade_file):
        print(f"\n✗ Cascade file not found: {cascade_file}")
        print("\nDOWNLOAD IT NOW:")
        print("cd ~/Desktop/plate_detection")
        print("curl -o haarcascade_russian_plate_number.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml")
        return
    
    # Initialize recognizer
    recognizer = RussianLicensePlateRecognizer()
    
    # Process images
    image_names = ["image1.jpg", "image2.jpg", "image3.jpg"]
    results = []
    
    for img_name in image_names:
        img_path = os.path.join(desktop, img_name)
        
        if not os.path.exists(img_path):
            print(f"\nImage not found: {img_path}")
            print(f"Please place {img_name} on your Desktop")
            continue
        
        result_img, plate_region = recognizer.process_image(img_path)
        
        if result_img is not None:
            cv2.imwrite(f"detected_{img_name}", result_img)
            results.append(result_img)
    
    # Display results if any
    if results:
        print("\n" + "="*60)
        print(f"Displaying {len(results)} results...")
        print("Press 'q' to quit, any other key to continue")
        print("="*60)
        
        for i, result_img in enumerate(results):
            cv2.imshow(f"Result {i+1}", result_img)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            print("Quitting...")
        
        cv2.destroyAllWindows()
    else:
        print("\n" + "="*60)
        print("NO PLATES DETECTED IN ANY IMAGES")
        print("="*60)
        print("\nTroubleshooting steps:")
        print("1. Make sure images contain Russian license plates")
        print("2. Check that plates are clearly visible (not too small/blurry)")
        print("3. Run the download command above if cascade missing")
        print("\nDebug images saved as 'debug_prep_*.jpg' - inspect these")
    
    print(f"\nAll files saved to: {project_dir}")

if __name__ == "__main__":
    main()