import cv2
import json

polygon_slots = []
current_polygon = []
frame = None


def mouse_callback(event, x, y, flags, param):
    global current_polygon, polygon_slots, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        print(f"Titik ke-{len(current_polygon)}: {x}, {y}")

        if len(current_polygon) == 4:
            polygon_slots.append(
                {"id": len(polygon_slots) + 1, "points": current_polygon.copy()}
            )
            current_polygon.clear()
            print(f"Slot ke-{len(polygon_slots)} disimpan.")


def draw_polygons(img):
    for slot in polygon_slots:
        pts = slot["points"]
        for i in range(len(pts)):
            cv2.line(img, pts[i], pts[(i + 1) % len(pts)], (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Slot {slot['id']}",
            pts[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )


def main(video_path, save_path="slot_polygons.json"):
    global frame

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari video.")
        return

    cv2.namedWindow("Label Slot Parkir")
    cv2.setMouseCallback("Label Slot Parkir", mouse_callback)

    while True:
        temp_frame = frame.copy()
        draw_polygons(temp_frame)

        # Gambar titik polygon yang sedang dibuat
        for pt in current_polygon:
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)

        cv2.imshow("Label Slot Parkir", temp_frame)
        key = cv2.waitKey(1)

        if key == ord("s"):
            with open(save_path, "w") as f:
                json.dump(polygon_slots, f, indent=2)
            print(f"Disimpan ke {save_path}")
            break
        elif key == ord("q"):
            print("Keluar tanpa menyimpan.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("./input6.mp4")
