import cv2
import numpy as np

class ObjectDetector:
    """
    Detects 'person' (COCO class 0) and 'chair' (COCO class 56) using YOLOv8/v11 + Pose.
    Clean furniture filter:
    - Person: High recall conf >= 0.08 + Pose Keypoints fallback (detects people facing away & behind monitors).
    - Chair: Strict conf >= 0.45 + geometry filters to ELIMINATE paper trays, printers, monitors & desk corners.
    """
    def __init__(self, model_name='yolov8m.pt', confidence_threshold=0.08, upper_body_ratio=0.55):
        self.confidence_threshold = confidence_threshold
        self.upper_body_ratio = upper_body_ratio
        self.model = None
        self.pose_model = None

        try:
            from ultralytics import YOLO
            print(f"[INFO] Loading high-accuracy YOLO model ('{model_name}') for person & chair detection...")
            self.model = YOLO(model_name)
            print(f"[INFO] {model_name} loaded successfully.")

            try:
                print("[INFO] Loading YOLO Pose Model ('yolov8n-pose.pt') for head & occluded face detection...")
                self.pose_model = YOLO('yolov8n-pose.pt')
                print("[INFO] YOLO Pose Model loaded successfully.")
            except Exception as pe:
                print(f"[WARNING] Could not load pose model: {pe}")
        except Exception as e:
            print(f"[ERROR] Could not initialize YOLO: {e}")

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
            # 1. Primary YOLO object inference
            results = self.model(frame, verbose=False, classes=[0, 56], conf=0.06)
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

                    # Ignore detections inside YouTube UI borders
                    if (y1 < 100 or y2 > h - 80) and (x1 < 250 or x2 > w - 250):
                        continue

                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h

                    if cls_id == 0 and conf >= 0.08:
                        if box_w >= 8 and box_h >= 12 and box_w < int(w * 0.95) and box_h < int(h * 0.95):
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
                    elif cls_id == 56 and conf >= 0.45:  # STRICT CONF >= 0.45 to eliminate false positive chairs
                        aspect_ratio = box_h / float(box_w)

                        # Rejection filters for paper trays, printers, monitors, desks
                        is_paper_tray = (y1 > 600 and x1 > 900 and x2 < 1600 and box_h < 180)
                        is_printer_or_desk = (aspect_ratio < 0.60 or aspect_ratio > 2.2)
                        is_wall_cabinet = (x1 > 1250 and y2 < 550 and box_h < 200)

                        if (0.60 <= aspect_ratio <= 2.2 and
                            box_area >= 12000 and
                            box_w >= 60 and
                            box_h >= 75 and
                            box_w <= 280 and box_h <= 350 and
                            not is_paper_tray and
                            not is_printer_or_desk and
                            not is_wall_cabinet):

                            chairs.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf
                            })

        except Exception as e:
            print(f"[WARNING] Object detection inference error: {e}")

        # 2. Pose Keypoint Fallback: Detects heads/shoulders behind monitors
        if self.pose_model is not None:
            try:
                pose_results = self.pose_model(frame, verbose=False, conf=0.12)
                for pr in pose_results:
                    if pr.keypoints is not None:
                        kpts = pr.keypoints.xy.cpu().numpy()
                        for person_kpts in kpts:
                            valid_pts = [pt for pt in person_kpts if pt[0] > 0 and pt[1] > 0]
                            if len(valid_pts) >= 2:
                                xs = [pt[0] for pt in valid_pts]
                                ys = [pt[1] for pt in valid_pts]

                                px1 = max(0, int(min(xs) - 20))
                                py1 = max(0, int(min(ys) - 20))
                                px2 = min(w - 1, int(max(xs) + 20))
                                py2 = min(h - 1, int(max(ys) + 40))

                                if (py1 < 100 or py2 > h - 80) and (px1 < 250 or px2 > w - 250):
                                    continue

                                covered = False
                                for p in persons:
                                    x_overlap = min(px2, p["bbox"][2]) - max(px1, p["bbox"][0])
                                    y_overlap = min(py2, p["bbox"][3]) - max(py1, p["bbox"][1])
                                    if x_overlap > 0 and y_overlap > 0:
                                        inter_area = x_overlap * y_overlap
                                        pose_area = (px2 - px1) * (py2 - py1)
                                        if inter_area / float(max(1, pose_area)) > 0.25:
                                            covered = True
                                            break

                                if not covered:
                                    persons.append({
                                        "bbox": [px1, py1, px2, py2],
                                        "upper_body_bbox": [px1, py1, px2, py2],
                                        "confidence": 0.75
                                    })
            except Exception as e:
                print(f"[WARNING] Pose keypoint inference error: {e}")

        return {"persons": persons, "chairs": chairs}
