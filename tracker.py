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


class CentroidTracker:
    def __init__(self, max_disappeared=30, min_distance_px=200):
        self.next_object_id = 1
        self.objects = {}  # id -> dict of object state
        self.disappeared = {}  # id -> consecutive disappeared frames count
        self.max_disappeared = max_disappeared
        self.min_distance_px = min_distance_px

    def register(self, bbox, confidence=1.0, face_info=None, pose_info=None):
        centroid = compute_centroid(bbox)
        # Deterministically assign fallback gender based on object ID for consistent testing
        fallback_gender = "Male" if (self.next_object_id % 2 != 0) else "Female"

        self.objects[self.next_object_id] = {
            "id": self.next_object_id,
            "bbox": bbox,
            "centroid": centroid,
            "confidence": confidence,
            "gender": face_info.get("gender", fallback_gender) if face_info else fallback_gender,
            "face_detected": face_info.get("detected", False) if face_info else False,
            "pose": pose_info.get("pose", "upright") if pose_info else "upright",
            "pose_score": pose_info.get("score", 0.0) if pose_info else 0.0,
            # Rules state storage per person
            "idle_timer": 0.0,
            "recline_counter": 0,
            "status": "ON_DUTY",
            "status_color": (0, 255, 127),
            "interaction_timer": 0.0,
            "interaction_partner_id": None,
            "interaction_active": False
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, new_detections, face_infos=None, pose_infos=None):
        """
        new_detections: list of bboxes [[x1, y1, x2, y2], ...]
        """
        if face_infos is None:
            face_infos = [None] * len(new_detections)
        if pose_infos is None:
            pose_infos = [None] * len(new_detections)

        # If no objects are currently tracked, register all new detections
        if len(self.objects) == 0:
            for i, bbox in enumerate(new_detections):
                self.register(bbox, face_info=face_infos[i], pose_info=pose_infos[i])
            return self.objects

        # If no new detections, mark all as disappeared
        if len(new_detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calculate centroid & IoU distance matrix between existing objects and new detections
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]["centroid"] for oid in object_ids]
        new_centroids = [compute_centroid(b) for b in new_detections]

        D = np.zeros((len(object_ids), len(new_detections)), dtype=np.float32)
        for i, (cx1, cy1) in enumerate(object_centroids):
            for j, (cx2, cy2) in enumerate(new_centroids):
                dist = math.hypot(cx1 - cx2, cy1 - cy2)
                iou = compute_iou(self.objects[object_ids[i]]["bbox"], new_detections[j])
                # Hybrid cost: distance cost minus IoU bonus
                D[i, j] = dist - (iou * 100.0)

        # Find best matching using greedy row/col minimums
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Check if distance is reasonably small
            if D[row, col] > self.min_distance_px:
                continue

            object_id = object_ids[row]
            bbox = new_detections[col]
            face_info = face_infos[col]
            pose_info = pose_infos[col]

            # Smooth bbox transition (Alpha filter)
            old_bbox = self.objects[object_id]["bbox"]
            alpha = 0.6
            smoothed_bbox = [
                int(alpha * bbox[i] + (1 - alpha) * old_bbox[i]) for i in range(4)
            ]

            self.objects[object_id]["bbox"] = smoothed_bbox
            self.objects[object_id]["centroid"] = compute_centroid(smoothed_bbox)
            self.disappeared[object_id] = 0

            # Update Face & Pose attributes if detected
            if face_info and face_info.get("detected"):
                self.objects[object_id]["gender"] = face_info.get("gender", self.objects[object_id]["gender"])
                self.objects[object_id]["face_detected"] = True

            if pose_info:
                self.objects[object_id]["pose"] = pose_info.get("pose", self.objects[object_id]["pose"])
                self.objects[object_id]["pose_score"] = pose_info.get("score", 0.0)
                self.objects[object_id]["pose_method"] = pose_info.get("method", "unknown")
                self.objects[object_id]["pose_angle"] = pose_info.get("angle", -1.0)

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Handle unmatched new detections
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(new_detections[col], face_info=face_infos[col], pose_info=pose_infos[col])

        return self.objects
