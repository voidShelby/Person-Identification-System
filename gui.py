import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import face_recognition
import joblib
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from eda import variance_of_laplacian

# Load trained model
knn = joblib.load("models/knn_classifier.pkl")

# Predict identity from image path
def identify_from_image(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(image)
        if not encs:
            return "⚠️ No face detected."
        encoding = encs[0]
        probs = knn.predict_proba([encoding])[0]
        pred = knn.predict([encoding])[0]
        conf = np.max(probs)

        if conf >= 0.75:
            return f"✅ Match: {pred} ({conf:.2f})"
        elif conf >= 0.55:
            return f"🤔 Similar to: {pred} ({conf:.2f})"
        else:
            return f"❌ Unknown ({conf:.2f})"
    except Exception as e:
        return f"Error: {str(e)}"

# Test image from testimage/ folder
def browse_and_predict():
    file_path = filedialog.askopenfilename(
        initialdir="testimage/",
        title="Select Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    if file_path:
        result = identify_from_image(file_path)
        messagebox.showinfo("Prediction Result", result)

# Upload image from anywhere
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Upload Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    if file_path:
        result = identify_from_image(file_path)
        messagebox.showinfo("Upload Image Result", result)

# Run real-time webcam-based identification
def run_webcam():
    os.system("python real_time_camera.py")

# Show EDA summary graph
def show_eda_results():
    from eda import DATASET_DIR
    person_ids, total, blurry, failed = [], [], [], []
    for pid in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, pid)
        if not os.path.isdir(path): continue
        person_ids.append(pid)
        count = 0; blur = 0; undetect = 0
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if variance_of_laplacian(gray) < 70:
                blur += 1
            face = face_recognition.face_locations(img)
            if len(face) == 0:
                undetect += 1
            count += 1
        total.append(count)
        blurry.append(blur)
        failed.append(undetect)

    plt.figure(figsize=(10, 5))
    plt.bar(person_ids, total, label="Total")
    plt.bar(person_ids, blurry, label="Blurry")
    plt.bar(person_ids, failed, label="Undetected")
    plt.title("EDA Results")
    plt.xlabel("Person ID")
    plt.ylabel("Image Count")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------- GUI Layout ----------
root = tk.Tk()
root.title("Person Identification System")
root.geometry("400x350")
root.resizable(False, False)

tk.Label(root, text="Select an Option", font=("Helvetica", 16)).pack(pady=20)

tk.Button(root, text="1️⃣ Identify From Testimage", command=browse_and_predict, width=30).pack(pady=8)
tk.Button(root, text="2️⃣ Upload Image to Identify", command=upload_and_predict, width=30).pack(pady=8)
tk.Button(root, text="3️⃣ Run Webcam Identification", command=run_webcam, width=30).pack(pady=8)
tk.Button(root, text="4️⃣ Show EDA Results", command=show_eda_results, width=30).pack(pady=8)

tk.Label(root, text="Press close to exit", font=("Helvetica", 10)).pack(pady=20)

root.mainloop()
