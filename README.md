# 🎨 Kolam Analyzer & Recreator

## 🧠 Overview
The **Kolam Analyzer** is a Python-based computer vision project that analyzes and recreates traditional Kolam (Rangoli) patterns.  
It detects dots, curves, and symmetry in Kolam designs and then reconstructs a scaled, mathematically derived version of the pattern.

---

## ⚙️ Features
- 🔍 **Automatic Dot Detection** using Hough Circle Transform  
- 🌀 **Contour Extraction** to isolate Kolam lines and curves  
- 🧾 **Grid Structure Estimation** using DBSCAN clustering  
- 🔄 **Symmetry Detection** to identify balanced patterns  
- 🧩 **Design Principle Extraction** (dot count, contour count, symmetry, grid size, etc.)  
- 🎭 **Kolam Recreation** using extracted geometry  
- 📊 **Visualization Dashboard** with detected dots, contours, and analysis summary  
- 💾 **JSON Export** of extracted design principles  

---
## 🧰 Requirements
Before running the project, install the required Python libraries:
```bash
pip install opencv-python numpy matplotlib scipy scikit-learn
```
🚀 Usage
Place your Kolam image (e.g., image5.png) in the project directory.
Run the analysis:
```
python3 main.py
```
## Outputs generated:
- 🖼️ kolam_analysis.png — visualization of detected dots, contours, and summary
- 🧠 kolam_principles.json — extracted design data
- 🎨 recreated_kolam.png — recreated pattern based on design principles
  
- 1️⃣ Image Loading
Reads the input Kolam image.
Converts it to grayscale and extracts height/width.
-2️⃣ Dot Detection
Uses Gaussian blur for noise reduction.
Detects circular dot patterns via cv2.HoughCircles().
-3️⃣ Contour Detection
Thresholds the grayscale image to a binary mask.
Removes detected dots to isolate the Kolam’s connecting lines.
Filters small/noisy contours.
-4️⃣ Grid & Symmetry Analysis
Determines dot alignment grid using DBSCAN clustering.
Checks for horizontal and vertical symmetry around the design’s center.

💡 Future Enhancements
-Add Flask / Streamlit Web Interface
-Enable real-time camera input
-Integrate Kolam classification (AI/ML)
-Export SVG vectorized Kolam recreations
