from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="pen/data.yaml",
    epochs=30,
    imgsz=416,     
    batch=8,
    workers=2,
    device="cpu",
)