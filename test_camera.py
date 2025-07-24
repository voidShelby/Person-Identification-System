import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to access the webcam.")
else:
    print("✅ Webcam opened successfully. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
