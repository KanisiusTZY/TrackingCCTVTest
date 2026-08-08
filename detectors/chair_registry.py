import math
import numpy as np
from tracker import compute_iou, compute_centroid

class ChairRegistry:
    """
    Persistent Chair Registry.
    Maintains a permanent list of known chair positions across the video stream.

    - Live YOLO chair detections with high confidence (>= min_confidence) update or register
      permanent chair entries in the registry.
    - Chairs are NEVER removed simply because they get occluded by a sitting person.
    - Positions are refined when high-confidence detections re-confirm them.
    """

    def __init__(self, min_confidence=0.50, match_iou_threshold=0.40):
        self.min_confidence = min_confidence
        self.match_iou_threshold = match_iou_threshold
        self.next_chair_id = 1
        self.registry = {}  # chair_id -> dict {"id": int, "bbox": [x1,y1,x2,y2], "name": str, "conf": float, "detection_count": int}

    def update(self, live_chair_detections):
        """
        Processes live chair detections from YOLO and updates the persistent registry.
        live_chair_detections: list of dicts [{"bbox": [x1,y1,x2,y2], "confidence": float}]
        """
        for det in live_chair_detections:
            conf = det.get("confidence", 1.0)
            if conf < self.min_confidence:
                continue

            bbox = det["bbox"]
            matched_id = None
            best_iou = 0.0

            # Try to match with existing registry entries
            for cid, entry in self.registry.items():
                iou = compute_iou(bbox, entry["bbox"])
                if iou > best_iou and iou >= self.match_iou_threshold:
                    best_iou = iou
                    matched_id = cid

            if matched_id is not None:
                # Refine position using smooth exponential moving average (alpha=0.20)
                old_bbox = self.registry[matched_id]["bbox"]
                alpha = 0.20
                smoothed = [
                    int(alpha * bbox[i] + (1 - alpha) * old_bbox[i]) for i in range(4)
                ]
                self.registry[matched_id]["bbox"] = smoothed
                self.registry[matched_id]["conf"] = max(conf, self.registry[matched_id]["conf"])
                self.registry[matched_id]["detection_count"] += 1
            else:
                # New permanent chair discovered — add to registry!
                cid = self.next_chair_id
                self.next_chair_id += 1
                self.registry[cid] = {
                    "id": cid,
                    "bbox": bbox,
                    "name": f"Chair {cid}",
                    "conf": conf,
                    "detection_count": 1
                }
                print(f"[CHAIR REGISTRY] Discovered & Registered New Chair #{cid} at BBox {bbox} (conf={conf:.2f})")

        return self.registry

    def get_all_chairs(self):
        """
        Returns all registered permanent chairs.
        """
        return self.registry
