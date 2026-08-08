import cv2
import numpy as np

class ObjectDetector:
    """
    Detects both 'person' (COCO class 0) and 'chair' (COCO class 56) using YOLOv8.
    Applies class-specific confidence, dimension, and aspect-ratio filters:
    - Person: conf >= 0.35, accommodates foreground & background employees.
    - Chair: conf >= 0.35, min area >= 15,000 px^2, filters out paper trays & desks.
    """
    def __init__(self, confidence_threshold=0.30, upper_body_ratio=0.55):
        self.confidence_threshold = confidence_threshold
        self.upper_body_ratio = upper_body_ratio
        self.model = None

        try:
            from ultralytics import YOLO
            print("[INFO] Loading YOLOv8 object detector model (person & chair)...")
            self.model = YOLO('yolov8n.pt')
            print("[INFO] YOLOv8 loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Could not initialize YOLOv8: {e}")

    def detect(self, frame, upper_body_ratio=None):
        if upper_body_ratio is not None:
            ratio = upper_body_ratio
        else:
            ratio = self.upper_body_ratio

        h, w = frame.shape[:2]
        persons = []
        chairs = []

        if self.model is None:
            return {"persons": persons, "chairs": chairs}

        try:
            results = self.model(frame, verbose=False, classes=[0, 56], conf=0.20)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h

                    if cls_id == 0 and conf >= 0.30:
                        # Person: accept both foreground and background persons
                        if box_w >= 20 and box_h >= 30 and box_w < int(w * 0.90) and box_h < int(h * 0.95):
                            y2_upper = y1 + int(box_h * ratio)
                            y2_upper = min(y2, max(y1 + 10, y2_upper))

                            persons.append({
                                "bbox": [x1, y1, x2, y2],
                                "upper_body_bbox": [x1, y1, x2, y2_upper],
                                "confidence": conf
                            })
                    elif cls_id == 56 and conf >= 0.35:
                        # Chair: filter invalid aspect ratios and small garbage objects (paper trays, monitor stands)
                        aspect_ratio = box_h / float(box_w)
                        if 0.5 <= aspect_ratio <= 2.2 and box_area >= 14000 and box_w >= 70 and box_h >= 80:
                            chairs.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf
                            })

        except Exception as e:
            print(f"[WARNING] Object detection inference error: {e}")

        return {"persons": persons, "chairs": chairs}
