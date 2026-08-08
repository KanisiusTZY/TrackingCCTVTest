import cv2
import numpy as np

class ObjectDetector:
    """
    Detects both 'person' (COCO class 0) and 'chair' (COCO class 56) using YOLOv8.
    For person detections, also computes upper-body bounding box based on upper_body_ratio.
    """
    def __init__(self, confidence_threshold=0.25, upper_body_ratio=0.55):
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
        """
        Detects persons and chairs in the frame.
        Returns dict:
        {
          "persons": [{"bbox": [x1,y1,x2,y2], "upper_body_bbox": [x1,y1,x2,y2_upper], "confidence": float}],
          "chairs": [{"bbox": [x1,y1,x2,y2], "confidence": float}]
        }
        """
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
            results = self.model(frame, verbose=False, classes=[0, 56], conf=self.confidence_threshold)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Clamp to frame boundary
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    if box_w <= 10 or box_h <= 10:
                        continue

                    if cls_id == 0:
                        # Person: compute upper body bbox
                        y2_upper = y1 + int(box_h * ratio)
                        y2_upper = min(y2, max(y1 + 10, y2_upper))

                        full_bbox = [x1, y1, x2, y2]
                        upper_bbox = [x1, y1, x2, y2_upper]

                        persons.append({
                            "bbox": full_bbox,
                            "upper_body_bbox": upper_bbox,
                            "confidence": conf
                        })
                    elif cls_id == 56:
                        # Chair: keep tight bounding box as predicted
                        chairs.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf
                        })

        except Exception as e:
            print(f"[WARNING] Object detection inference error: {e}")

        return {"persons": persons, "chairs": chairs}
