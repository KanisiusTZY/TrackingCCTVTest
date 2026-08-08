import cv2
import numpy as np

class ObjectDetector:
    """
    Detects 'person' (COCO class 0) and 'chair' (COCO class 56) using YOLOv8.
    Applies strict geometry and spatial filters to eliminate wall cabinets, desk drawers, and standing-person false positives.
    """
    def __init__(self, confidence_threshold=0.20, upper_body_ratio=0.55):
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
            results = self.model(frame, verbose=False, classes=[0, 56], conf=0.18)
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

                    if cls_id == 0 and conf >= 0.20:
                        if box_w >= 15 and box_h >= 25 and box_w < int(w * 0.95) and box_h < int(h * 0.98):
                            y2_upper = y1 + int(box_h * ratio)
                            y2_upper = min(y2, max(y1 + 10, y2_upper))

                            margin_x = int(box_w * 0.08)
                            x1_upper = x1 + margin_x
                            x2_upper = x2 - margin_x

                            persons.append({
                                "bbox": [x1, y1, x2, y2],
                                "upper_body_bbox": [x1_upper, y1, x2_upper, y2_upper],
                                "confidence": conf
                            })
                    elif cls_id == 56 and conf >= 0.40:
                        # Chair filter:
                        # 1. Require real chair dimensions (conf >= 0.40, area >= 14000, h >= 80, w >= 70)
                        # 2. Rejection for wall cabinets under window (x1 > 1200 and y2 < 680 and h < 350)
                        # 3. Spatial rejection for paper trays & desk surfaces (y1 > 640 and x1 > 980)
                        aspect_ratio = box_h / float(box_w)
                        is_paper_tray_or_desk = (y1 > 640 and x1 > 980 and x2 < 1600)
                        is_wall_cabinet = (x1 > 1200 and y2 < 680 and box_h < 360)
                        is_flat_desk = (aspect_ratio < 0.65 and y1 > 400)

                        if (0.60 <= aspect_ratio <= 2.2 and
                            box_area >= 14000 and
                            box_w >= 70 and
                            box_h >= 80 and
                            not is_paper_tray_or_desk and
                            not is_wall_cabinet and
                            not is_flat_desk):

                            chairs.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf
                            })

        except Exception as e:
            print(f"[WARNING] Object detection inference error: {e}")

        return {"persons": persons, "chairs": chairs}
