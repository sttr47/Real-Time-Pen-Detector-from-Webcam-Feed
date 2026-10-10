import cv2
import time
from ultralytics import YOLO

model_path = "runs/detect/train3/weights/best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)   
conf = 0.6
fps_smooth = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("No frame")
        break

    t0 = time.time()
    results = model(frame, conf=conf, imgsz=416, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"pen {score:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    fps = 1 / max(time.time() - t0, 1e-6)
    fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps
    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("Pen detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()