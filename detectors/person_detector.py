import cv2
import numpy as np
import math

def compute_centroid(bbox):
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]

def detect_active_video_roi(frame):
    """
    Dynamically detects the active video content viewport inside any frame.
    Works generalizably for raw CCTV feeds, letterboxed videos, and screen recordings.
    Returns (x_min, y_min, x_max, y_max).
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold dark letterbox/pillarbox regions
    _, thresh = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest non-dark contiguous region
        largest = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(largest)

        # If detected ROI spans at least 40% of frame area, treat as active video region
        if rw * rh >= (w * h * 0.40):
            # Exclude bottom player bar / comments strip if present (bottom 6% of video)
            return (rx, ry, rx + rw, ry + int(rh * 0.94))

    return (0, 0, w, h)

class ObjectDetector:
    """
    Generalizable Object & Pose Detector for Persons and Chairs.
    Works on any CCTV video stream or live camera feed without hardcoded coordinates.
    """
    def __init__(self, model_name='yolov8m.pt', confidence_threshold=0.10, upper_body_ratio=0.55):
        self.confidence_threshold = confidence_threshold
        self.upper_body_ratio = upper_body_ratio
        self.model = None
        self.pose_model = None

        try:
            from ultralytics import YOLO
            print(f"[INFO] Loading YOLO model ('{model_name}')...")
            self.model = YOLO(model_name)
            print(f"[INFO] {model_name} loaded successfully.")

            try:
                print("[INFO] Loading YOLO Pose Model ('yolov8n-pose.pt')...")
                self.pose_model = YOLO('yolov8n-pose.pt')
                print("[INFO] YOLO Pose Model loaded successfully.")
            except Exception as pe:
                print(f"[WARNING] Could not load pose model: {pe}")
        except Exception as e:
            print(f"[ERROR] Could not initialize YOLO: {e}")

    def detect(self, frame, upper_body_ratio=None):
        ratio = upper_body_ratio if upper_body_ratio is not None else self.upper_body_ratio
        h, w = frame.shape[:2]
        persons = []
        chairs = []

        if self.model is None:
            return {"persons": persons, "chairs": chairs}

        # Dynamically detect active video content ROI
        roi_x1, roi_y1, roi_x2, roi_y2 = detect_active_video_roi(frame)

        try:
            # 1. Primary YOLO object inference
            results = self.model(frame, verbose=False, classes=[0, 56], conf=0.10)
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

                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    # Reject detections outside dynamic active video ROI
                    if cx < roi_x1 or cx > roi_x2 or cy < roi_y1 or cy > roi_y2:
                        continue

                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h

                    # Ignore extreme edge noise / full-screen artifacts
                    if box_w < 15 or box_h < 20 or box_w > int(w * 0.95) or box_h > int(h * 0.95):
                        continue

                    if cls_id == 0 and conf >= 0.08:  # Person
                        y2_upper = y1 + int(box_h * max(ratio, 0.55))
                        y2_upper = min(y2, max(y1 + 15, y2_upper))

                        margin_x = int(box_w * 0.04)
                        x1_upper = max(0, x1 - margin_x)
                        x2_upper = min(w - 1, x2 + margin_x)

                        persons.append({
                            "bbox": [x1, y1, x2, y2],
                            "upper_body_bbox": [x1_upper, y1, x2_upper, y2_upper],
                            "confidence": conf
                        })

                    elif cls_id == 56 and conf >= 0.50:  # Physical Chair (high confidence only)
                        aspect_ratio = box_h / float(max(1, box_w))
                        if (0.55 <= aspect_ratio <= 2.2 and
                            box_area >= 4000 and
                            60 <= box_w <= 320 and
                            70 <= box_h <= 380):
                            chairs.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf
                            })

        except Exception as e:
            print(f"[WARNING] Object detection inference error: {e}")

        # 2. Keypoint Pose Assist for Occluded Persons behind monitors
        if self.pose_model is not None:
            try:
                pose_results = self.pose_model(frame, verbose=False, conf=0.10)
                for pr in pose_results:
                    if pr.keypoints is not None and len(pr.keypoints.xy) > 0:
                        kpts_all = pr.keypoints.xy.cpu().numpy()
                        confs_all = pr.keypoints.conf.cpu().numpy() if pr.keypoints.conf is not None else None

                        for idx, person_kpts in enumerate(kpts_all):
                            p_confs = confs_all[idx] if confs_all is not None else np.ones(len(person_kpts))
                            valid_pts = [person_kpts[i] for i in range(len(person_kpts))
                                         if p_confs[i] >= 0.15 and person_kpts[i][0] > 0 and person_kpts[i][1] > 0]

                            if len(valid_pts) >= 2:
                                xs = [pt[0] for pt in valid_pts]
                                ys = [pt[1] for pt in valid_pts]

                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)

                                head_cx = (min_x + max_x) / 2.0
                                head_cy = (min_y + max_y) / 2.0

                                # Reject pose detections outside dynamic active video ROI
                                if head_cx < roi_x1 or head_cx > roi_x2 or head_cy < roi_y1 or head_cy > roi_y2:
                                    continue

                                kpt_w = max(35.0, max_x - min_x)
                                kpt_h = max(35.0, max_y - min_y)

                                box_w_synth = max(90, int(kpt_w * 1.8))
                                box_h_synth = max(110, int(kpt_h * 2.2))

                                px1 = max(0, int(head_cx - box_w_synth / 2.0))
                                py1 = max(0, int(min_y - 15))
                                px2 = min(w - 1, int(head_cx + box_w_synth / 2.0))
                                py2 = min(h - 1, int(py1 + box_h_synth))

                                covered = False
                                for p in persons:
                                    pc = compute_centroid(p["bbox"])
                                    dist = math.hypot(head_cx - pc[0], head_cy - pc[1])
                                    if dist < 75.0:
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
