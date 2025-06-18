import cv2
import json
import numpy as np

# Load titik-titik slot parkir dari JSON
with open("anotasi/slot_polygons5.json") as f:
    slots = json.load(f)  # Ini list of dicts

cap = cv2.VideoCapture("input5.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for slot in slots:
        points = slot["points"]  # List of [x, y]
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Label pakai id slot
        cv2.putText(
            frame,
            f"Slot {slot['id']}",
            tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Parking Slots Overlay", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
