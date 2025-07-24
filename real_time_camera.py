import cv2
import face_recognition
import numpy as np
import joblib

model_path = "models/knn_classifier.pkl"
knn = joblib.load(model_path)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def get_label(pred, conf):
    if conf >= 0.75:
        return f"✅ Match: {pred}", (0, 255, 0)
    elif conf >= 0.55:
        return f"🤔 Similar to: {pred}", (0, 165, 255)
    else:
        return "❌ Unknown", (0, 0, 255)

cap = cv2.VideoCapture(0)
print("📸 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    processed = preprocess(frame)
    small = cv2.resize(processed, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        probs = knn.predict_proba([face_enc])[0]
        pred = knn.predict([face_enc])[0]
        conf = np.max(probs)

        label, color = get_label(pred, conf)

        # scale back up
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (left, bottom + 10), (left + int(conf * 200), bottom + 25), color, -1)

    cv2.imshow("Real-Time Person Identification (Robust Mode)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
