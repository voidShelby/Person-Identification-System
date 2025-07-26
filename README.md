# Person Identification System

This project implements a real-time **Person Identification System** using facial features. It emphasizes **interpretable machine learning** by applying **Computational Thinking (CT)** principles. Developed as part of the *Introduction to Computing* course at IBA Karachi.

## Course Information
- Course: Introduction to Computing  
- Instructor: Prof. Dr. Rizwan Ahmed Khan  
- Institution: School of Mathematics and Computer Science, IBA Karachi  
- Submission Date: 26 July 2025  

## Project Objectives
- Design and implement a facial recognition system
- Apply CT principles: Decomposition, Pattern Recognition, Abstraction, and Algorithmic Thinking
- Build a transparent, interpretable classification system (using KNN)
- Analyze classifier decisions through visual and statistical explanation
- Provide a user-friendly desktop GUI

## Project Structure
PersonIdentificationSystem/
├── dataset/ # Training data (per-person subfolders)
├── testimage/ # Test image directory
├── models/ # Saved model files (knn_classifier.pkl, embeddings)
├── main.py # Model training script
├── gui.py # Tkinter-based GUI
├── eda.py # EDA logic and image quality checking
├── real_time_camera.py # Webcam recognition script
├── test_image_identification.py
├── requirements.txt
├── README.md
└── report.pdf 


 Installation Instructions
 1. Clone the repository
```bash
git clone https://github.com/voidShelby/Person-Identification-System.git
cd Person-Identification-System

2. Create a virtual environment (recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate     # For Windows
# OR
source venv/bin/activate  # For macOS/Linux

3. Install required packages
bash
Copy
Edit
pip install -r requirements.txt
pip install git+https://github.com/ageitgey/face_recognition_models
Ensure Python 3.10 or later is used.

Running the System
Step 1: Train the model (if not already trained)
bash
Copy
Edit
python main.py
This generates the knn_classifier.pkl file inside the models/ folder.


Step 2: Launch the GUI
bash
Copy
Edit
python gui.py


The GUI includes options to:
Test using an image from the testimage/ folder
Upload an image from your computer
Identify using your system’s webcam
Visualize EDA (blurry images, undetected faces, etc.)


Features and Interpretability:
Uses K-Nearest Neighbors classifier with face embeddings (128D vectors)
Identifies and explains predictions using confidence thresholds:
High confidence: "Match"
Medium confidence: "Similar"
Low confidence: "Unknown"
Displays EDA graphs:
Total images per person
Number of blurry images
Faces not detected
Analysis done using variance of Laplacian and face detection checks


Accuracy Metrics
Metric	Result
Test Image Accuracy	> 90%
Webcam Accuracy (varies)
Confidence Thresholds	0.75 (match), 0.55 (similar)



Technologies Used
Python 3.10+
face_recognition
OpenCV
tkinter
scikit-learn
joblib
matplotlib, seaborn (EDA)


Computational Thinking Application
Decomposition:	Modular scripts (GUI, model, camera, EDA)
Pattern Recognition:	Similar embeddings, blurry image detection
Abstraction:	Faces → 128D embeddings
Algorithmic Thinking:	Custom classifier logic, threshold logic, EDA statistics







