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


class SingleCategoryTracker:
    def __init__(self, category_prefix="obj", max_disappeared=30, min_distance_px=200):
        self.next_id = 1
        self.category_prefix = category_prefix
        self.objects = {}  # id -> dict of object state
        self.disappeared = {}  # id -> consecutive disappeared frames count
        self.max_disappeared = max_disappeared
        self.min_distance_px = min_distance_px

    def register(self, detection):
        bbox = detection["bbox"]
        centroid = compute_centroid(bbox)

        obj_data = {
            "id": self.next_id,
            "bbox": bbox,
            "centroid": centroid,
            "confidence": detection.get("confidence", 1.0),
        }
        if "upper_body_bbox" in detection:
            obj_data["upper_body_bbox"] = detection["upper_body_bbox"]

        self.objects[self.next_id] = obj_data
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, detections):
        """
        detections: list of dicts [{"bbox": [...], "upper_body_bbox": [...], "confidence": ...}]
        """
        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
            return self.objects

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]["centroid"] for oid in object_ids]
        new_centroids = [compute_centroid(d["bbox"]) for d in detections]

        D = np.zeros((len(object_ids), len(detections)), dtype=np.float32)
        for i, (cx1, cy1) in enumerate(object_centroids):
            for j, (cx2, cy2) in enumerate(new_centroids):
                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                iou = compute_iou(self.objects[object_ids[i]]["bbox"], detections[j]["bbox"])
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
            det = detections[col]
            bbox = det["bbox"]

            old_bbox = self.objects[object_id]["bbox"]
            alpha = 0.65
            smoothed_bbox = [
                int(alpha * bbox[i] + (1 - alpha) * old_bbox[i]) for i in range(4)
            ]

            self.objects[object_id]["bbox"] = smoothed_bbox
            self.objects[object_id]["centroid"] = compute_centroid(smoothed_bbox)
            self.objects[object_id]["confidence"] = det.get("confidence", 1.0)
            self.disappeared[object_id] = 0

            if "upper_body_bbox" in det:
                ub = det["upper_body_bbox"]
                if "upper_body_bbox" in self.objects[object_id]:
                    old_ub = self.objects[object_id]["upper_body_bbox"]
                    smoothed_ub = [
                        int(alpha * ub[i] + (1 - alpha) * old_ub[i]) for i in range(4)
                    ]
                    self.objects[object_id]["upper_body_bbox"] = smoothed_ub
                else:
                    self.objects[object_id]["upper_body_bbox"] = ub

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
            self.register(detections[col])

        return self.objects


class MultiCategoryTracker:
    """
    Maintains separate trackers for persons and detected chairs.
    Chairs use max_disappeared=10 (last-known bbox persistence for occlusion).
    Persons use max_disappeared=30.
    """
    def __init__(self, person_max_disappeared=30, chair_max_disappeared=10):
        self.person_tracker = SingleCategoryTracker(category_prefix="person", max_disappeared=person_max_disappeared)
        self.chair_tracker = SingleCategoryTracker(category_prefix="chair", max_disappeared=chair_max_disappeared)

    def update(self, person_detections, chair_detections):
        tracked_persons = self.person_tracker.update(person_detections)
        tracked_chairs = self.chair_tracker.update(chair_detections)

        return {
            "persons": tracked_persons,
            "chairs": tracked_chairs
        }
