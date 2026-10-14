# 🚗 Russian License Plate Detector

An automatic license plate detection and recognition system using OpenCV's Haar cascade classifiers and Tesseract OCR, specifically trained for Russian license plates.

## 📋 Project Overview

This project implements a complete pipeline for detecting and recognizing Russian license plates from images. The system uses:

- **Haar Cascade Classifier** for plate detection (trained specifically for Russian plates)
- **Image Preprocessing** (CLAHE, Gaussian blur, adaptive thresholding)
- **Skew Correction** using minimum area rectangle detection
- **Tesseract OCR** for character recognition with Russian language support

The system processes three test images, detects license plate regions with bounding boxes, and extracts alphanumeric characters after appropriate preprocessing.

## 🏗️ Project Structure

## 🚀 How to Run

### Prerequisites
- Python 3.7+
- OpenCV 4.x
- Tesseract OCR
- NumPy
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/KelseySemien/russian-license-plate-detector.git
cd russian-license-plate-detector

# Install dependencies
pip install opencv-python opencv-contrib-python numpy matplotlib

# Install Tesseract OCR (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Install Tesseract OCR (Linux/macOS)
sudo apt install tesseract-ocr tesseract-ocr-rus  # Linux
brew install tesseract tesseract-lang  # macOS

🧠 How It Works
1. Image Preprocessing
Grayscale Conversion: Reduces complexity

CLAHE (Contrast Limited Adaptive Histogram Equalization): Improves feature visibility in non-uniform lighting

Gaussian Blurring: Reduces high-frequency noise

2. License Plate Detection
Haar Cascade Classifier: Scans the image at multiple scales

Parameter Tuning:

scaleFactor: 1.05 to 1.2 (controls detection sensitivity)

minNeighbors: 3 to 5 (controls false positive reduction)

3. Skew Correction
Contour Detection: Finds all contours within the plate region

Convex Hull: Computes the minimum area rectangle

Rotation: Aligns the plate horizontally

4. Character Recognition
Adaptive Thresholding: Binarizes the plate image

Morphological Closing: Connects broken characters

Tesseract OCR: Recognizes characters with Russian language support

📊 Results
Example Output
Input Image	Detected Plate	
image1.jpg	✅ Detected	
image2.jpg	✅ Detected	
image3.jpg	✅ Detected	


🛠️ Technologies Used
OpenCV 4.x - Image processing and cascade classification

Tesseract OCR - Character recognition

NumPy - Numerical computations

Matplotlib - Visualization

⚠️ Challenges and Solutions
Challenge	Solution
Non-uniform lighting	CLAHE preprocessing
Skewed plates	Minimum area rectangle detection and rotation
Broken characters	Morphological closing
False positives	Cascade parameter tuning (minNeighbors, scaleFactor)
Russian characters	Tesseract Russian language support (-l rus)
🔮 Future Improvements
Deep Learning Detector: Replace Haar cascade with YOLO or SSD for better generalization

Real-time Processing: Integrate with video streams

Multiple Plates: Detect all plates in a single image

Improved OCR: Use Tesseract's LSTM engine with custom training

