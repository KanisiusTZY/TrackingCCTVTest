import cv2
import numpy as np

class PersonDetector:
    def __init__(self, confidence_threshold=0.35, use_yolo=True):
        self.confidence_threshold = confidence_threshold
        self.use_yolo = use_yolo
        self.model = None

        if self.use_yolo:
            try:
                from ultralytics import YOLO
                print("[INFO] Loading YOLOv8 person detector model...")
                self.model = YOLO('yolov8n.pt')
                print("[INFO] YOLOv8 loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Could not initialize YOLOv8: {e}. Falling back to OpenCV HOG Detector.")
                self.use_yolo = False

        if not self.use_yolo:
            # Fallback OpenCV HOG Person Detector
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """
        Detect persons in the frame.
        Returns: list of bounding boxes [[x1, y1, x2, y2], ...]
        """
        h, w = frame.shape[:2]
        bboxes = []

        if self.use_yolo and self.model is not None:
            try:
                results = self.model(frame, verbose=False, classes=[0], conf=self.confidence_threshold)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # Clamp coordinates within frame boundaries
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(0, min(w - 1, x2))
                        y2 = max(0, min(h - 1, y2))
                        if (x2 - x1) > 20 and (y2 - y1) > 20:
                            bboxes.append([x1, y1, x2, y2])
                return bboxes
            except Exception as e:
                print(f"[WARNING] YOLO inference error: {e}. Falling back to HOG.")

        # OpenCV HOG Fallback
        rects, weights = self.hog.detectMultiScale(
            frame, 
            winStride=(8, 8), 
            padding=(4, 4), 
            scale=1.05
        )
        for (x, y, bw, bh) in rects:
            bboxes.append([x, y, x + bw, y + bh])

        return bboxes
