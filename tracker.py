import math
import numpy as np

def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    intersection = inter_width * inter_height

    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class PersonTracker:
    """
    Centroid & IoU Multi-Object Tracker for Persons.
    Matches person full-body bboxes for persistent person_id tracking.
    Propagates upper_body_bbox for IoU matching against ChairRegistry.
    """
    def __init__(self, max_disappeared=30, min_distance_px=200):
        self.next_id = 1
        self.objects = {}  # id -> dict of person state
        self.disappeared = {}  # id -> consecutive disappeared frames count
        self.centroid_history = {}  # id -> list of recent centroids (max 20 frames)
        self.max_disappeared = max_disappeared
        self.min_distance_px = min_distance_px
        self.HISTORY_LEN = 20  # Number of frames to track for net displacement

    def register(self, detection):
        bbox = detection["bbox"]
        centroid = compute_centroid(bbox)

        obj_data = {
            "id": self.next_id,
            "bbox": bbox,
            "upper_body_bbox": detection.get("upper_body_bbox", bbox),
            "centroid": centroid,
            "confidence": detection.get("confidence", 1.0),
            "net_displacement": 0.0,  # Net distance from first observed position
        }

        self.objects[self.next_id] = obj_data
        self.disappeared[self.next_id] = 0
        self.centroid_history[self.next_id] = [centroid]
        self.next_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.centroid_history:
            del self.centroid_history[object_id]

    def update(self, person_detections):
        """
        person_detections: list of dicts [{"bbox": [...], "upper_body_bbox": [...], "confidence": ...}]
        """
        if len(self.objects) == 0:
            for det in person_detections:
                self.register(det)
            return self.objects

        if len(person_detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]["centroid"] for oid in object_ids]
        new_centroids = [compute_centroid(d["bbox"]) for d in person_detections]

        D = np.zeros((len(object_ids), len(person_detections)), dtype=np.float32)
        for i, (cx1, cy1) in enumerate(object_centroids):
            for j, (cx2, cy2) in enumerate(new_centroids):
                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                iou = compute_iou(self.objects[object_ids[i]]["bbox"], person_detections[j]["bbox"])
                D[i, j] = dist - (iou * 100.0)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.min_distance_px:
                continue

            object_id = object_ids[row]
            det = person_detections[col]
            bbox = det["bbox"]

            old_bbox = self.objects[object_id]["bbox"]
            alpha = 0.65
            smoothed_bbox = [
                int(alpha * bbox[i] + (1 - alpha) * old_bbox[i]) for i in range(4)
            ]

            self.objects[object_id]["bbox"] = smoothed_bbox
            new_centroid = compute_centroid(smoothed_bbox)
            self.objects[object_id]["centroid"] = new_centroid
            self.objects[object_id]["confidence"] = det.get("confidence", 1.0)
            self.disappeared[object_id] = 0

            # Update centroid history for net displacement calculation
            history = self.centroid_history.get(object_id, [])
            history.append(new_centroid)
            if len(history) > self.HISTORY_LEN:
                history = history[-self.HISTORY_LEN:]
            self.centroid_history[object_id] = history

            # Net displacement = distance between oldest and newest centroid in history
            if len(history) >= 2:
                oldest = history[0]
                newest = history[-1]
                net_disp = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
            else:
                net_disp = 0.0
            self.objects[object_id]["net_displacement"] = net_disp

            ub = det.get("upper_body_bbox", bbox)
            old_ub = self.objects[object_id].get("upper_body_bbox", ub)
            smoothed_ub = [
                int(alpha * ub[i] + (1 - alpha) * old_ub[i]) for i in range(4)
            ]
            self.objects[object_id]["upper_body_bbox"] = smoothed_ub

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(person_detections[col])

        return self.objects
