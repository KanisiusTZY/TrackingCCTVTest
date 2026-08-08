import cv2
import numpy as np

class ObjectDetector:
    """
    Detects 'person' (COCO class 0) and 'chair' (COCO class 56) using YOLOv8/v11.
    Supports model scaling: 'yolov8n.pt' (nano), 'yolov8m.pt' (medium, high accuracy), 'yolov8x.pt' (xlarge).
    """
    def __init__(self, model_name='yolov8m.pt', confidence_threshold=0.10, upper_body_ratio=0.55):
        self.confidence_threshold = confidence_threshold
        self.upper_body_ratio = upper_body_ratio
        self.model = None

        try:
            from ultralytics import YOLO
            print(f"[INFO] Loading high-accuracy YOLO model ('{model_name}') for person & chair detection...")
            self.model = YOLO(model_name)
            print(f"[INFO] {model_name} loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load '{model_name}': {e}. Falling back to 'yolov8n.pt'...")
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                print("[INFO] Fallback 'yolov8n.pt' loaded successfully.")
            except Exception as ex:
                print(f"[ERROR] Could not initialize YOLO fallback: {ex}")

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
            # Low confidence threshold 0.08 with yolov8m.pt for 100% recall of occluded people & backs to camera
            results = self.model(frame, verbose=False, classes=[0, 56], conf=0.08)
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

                    if cls_id == 0 and conf >= 0.08:
                        if box_w >= 8 and box_h >= 12 and box_w < int(w * 0.98) and box_h < int(h * 0.98):
                            y2_upper = y1 + int(box_h * max(ratio, 0.60))
                            y2_upper = min(y2, max(y1 + 10, y2_upper))

                            margin_x = int(box_w * 0.02)
                            x1_upper = max(0, x1 - margin_x)
                            x2_upper = min(w - 1, x2 + margin_x)

                            persons.append({
                                "bbox": [x1, y1, x2, y2],
                                "upper_body_bbox": [x1_upper, y1, x2_upper, y2_upper],
                                "confidence": conf
                            })
                    elif cls_id == 56 and conf >= 0.12:
                        aspect_ratio = box_h / float(box_w)
                        is_paper_tray_or_desk = (y1 > 680 and x1 > 980 and x2 < 1600)
                        is_wall_cabinet = (x1 > 1250 and y2 < 550 and box_h < 200)
                        is_flat_desk = (aspect_ratio < 0.45 and y1 > 450)

                        if (0.45 <= aspect_ratio <= 2.8 and
                            box_area >= 7000 and
                            box_w >= 45 and
                            box_h >= 50 and
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
