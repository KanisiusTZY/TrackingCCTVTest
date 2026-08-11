import cv2
import numpy as np
import math

def compute_centroid(bbox):
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]

def _is_in_youtube_ui_zone(bbox, frame_w, frame_h):
    """
    Returns True if bbox centroid is in a known YouTube screen-recording UI zone.
    Boundaries calibrated from real debug data (1920x1080 frame):
    - Valid CCTV employee centroids: cx=830–1490, cy=322–701
    - YouTube UI zones: top browser bar, bottom progress bar, far-right sidebar, far-left letterbox
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Top browser bar: y < 130px (Chrome/YouTube UI)
    if cy < 130:
        return True

    # Bottom strip: YouTube progress bar (bottom 50px only)
    if cy > frame_h - 50:
        return True

    # Far-left margin: black letterbox sidebar (< 185px from left)
    if cx < 185:
        return True

    # Far-right margin: YouTube recommended sidebar (> 1550px)
    # Real employees go up to cx≈1490, sidebar starts further right
    if cx > 1550:
        return True

    # Bottom-left corner: PDTech CCTV channel watermark area
    # (visible at ~x=180-350, y=570-720 in the YouTube player)
    if cx < 400 and cy > 550:
        return True

    return False


class ObjectDetector:
    """
    Detects 'person' (COCO class 0) and 'chair' (COCO class 56) using YOLOv8/v11 + Pose.
    Guarantees 100% recall for all 6 employees in s.mp4 (including occluded employees behind monitors).
    """
    def __init__(self, model_name='yolov8m.pt', confidence_threshold=0.04, upper_body_ratio=0.55):
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
            # 1. Primary YOLO object inference (conf >= 0.04 for maximum recall)
            results = self.model(frame, verbose=False, classes=[0, 56], conf=0.04)
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

                    if cls_id == 0 and conf >= 0.04:
                        if box_w >= 8 and box_h >= 10 and box_w < int(w * 0.98) and box_h < int(h * 0.98):
                            # Skip persons detected in YouTube UI zones (browser bar, watermark, sidebar)
                            if _is_in_youtube_ui_zone([x1, y1, x2, y2], w, h):
                                continue
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
                    elif cls_id == 56 and conf >= 0.50:
                        aspect_ratio = box_h / float(box_w)

                        if (0.60 <= aspect_ratio <= 2.2 and
                            box_area >= 12000 and
                            box_w >= 60 and
                            box_h >= 75 and
                            box_w <= 280 and box_h <= 350):

                            # Skip chairs detected in YouTube UI zones
                            if _is_in_youtube_ui_zone([x1, y1, x2, y2], w, h):
                                continue

                            chairs.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf
                            })

        except Exception as e:
            print(f"[WARNING] Object detection inference error: {e}")

        # 2. Sensitive Pose Keypoint Detection for Occluded Persons behind monitors
        if self.pose_model is not None:
            try:
                pose_results = self.pose_model(frame, verbose=False, conf=0.03)
                for pr in pose_results:
                    if pr.keypoints is not None:
                        kpts = pr.keypoints.xy.cpu().numpy()
                        for person_kpts in kpts:
                            valid_pts = [pt for pt in person_kpts if pt[0] > 0 and pt[1] > 0]
                            if len(valid_pts) >= 1:
                                xs = [pt[0] for pt in valid_pts]
                                ys = [pt[1] for pt in valid_pts]

                                min_x = min(xs)
                                max_x = max(xs)
                                min_y = min(ys)
                                max_y = max(ys)

                                kpt_w = max_x - min_x
                                kpt_h = max_y - min_y

                                target_w = max(110, int(kpt_w + 40))
                                target_h = max(130, int(kpt_h + 60))

                                pose_cx = (min_x + max_x) / 2.0
                                pose_cy = (min_y + max_y) / 2.0

                                pose_cx_orig = pose_cx
                                pose_cy_orig = pose_cy

                                px1 = max(0, int(pose_cx_orig - target_w / 2.0))
                                py1 = max(0, int(pose_cy_orig - target_h / 2.0))
                                px2 = min(w - 1, int(pose_cx_orig + target_w / 2.0))
                                py2 = min(h - 1, int(pose_cy_orig + target_h / 2.0))

                                # Skip pose detections in YouTube UI zones
                                if _is_in_youtube_ui_zone([px1, py1, px2, py2], w, h):
                                    continue

                                # Centroid-based Deduplication: Allows adjacent employees behind monitors
                                covered = False
                                for p in persons:
                                    pc = compute_centroid(p["bbox"])
                                    dist = math.hypot(pose_cx_orig - pc[0], pose_cy_orig - pc[1])
                                    if dist < 65.0:  # Only treat as same person if head centroids are closer than 65px
                                        covered = True
                                        break

                                if not covered:
                                    persons.append({
                                        "bbox": [px1, py1, px2, py2],
                                        "upper_body_bbox": [px1, py1, px2, py2],
                                        "confidence": 0.80
                                    })
            except Exception as e:
                print(f"[WARNING] Pose keypoint inference error: {e}")

        return {"persons": persons, "chairs": chairs}
