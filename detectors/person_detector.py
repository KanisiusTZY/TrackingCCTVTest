import cv2
import numpy as np

class ObjectDetector:
    """
    Detects both 'person' (COCO class 0) and 'chair' (COCO class 56) using YOLOv8.
    Applies class-specific confidence thresholds:
    - Person: conf >= 0.35 (stable person detection)
    - Chair: conf >= 0.40 (filters out false-positive desks, tables, and cabinets)
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
            # Detect class 0 (person) and class 56 (chair)
            results = self.model(frame, verbose=False, classes=[0, 56], conf=0.25)
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

                    if box_w <= 15 or box_h <= 15:
                        continue

                    # Filter out oversized false positive boxes (e.g. whole room or huge table)
                    if box_w > int(w * 0.5) or box_h > int(h * 0.7):
                        continue

                    if cls_id == 0 and conf >= 0.35:
                        # Person: compute upper body bbox
                        y2_upper = y1 + int(box_h * ratio)
                        y2_upper = min(y2, max(y1 + 10, y2_upper))

                        persons.append({
                            "bbox": [x1, y1, x2, y2],
                            "upper_body_bbox": [x1, y1, x2, y2_upper],
                            "confidence": conf
                        })
                    elif cls_id == 56 and conf >= 0.40:
                        # Chair: filter invalid aspect ratios (eliminate long horizontal tables)
                        aspect_ratio = box_h / float(box_w)
                        if 0.5 <= aspect_ratio <= 2.2:
                            chairs.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf
                            })

        except Exception as e:
            print(f"[WARNING] Object detection inference error: {e}")

        return {"persons": persons, "chairs": chairs}
