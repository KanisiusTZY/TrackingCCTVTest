import math
import numpy as np
from tracker import compute_iou, compute_centroid

class ChairRegistry:
    """
    Persistent Chair Registry with Deduplication and Person-Bootstrap Fallback.

    Features:
    1. Strict Deduplication: Before adding a new chair entry, checks IoU against all existing
       registry entries (IoU >= 0.35). If matched, updates position without spawning duplicate entries.
    2. Auto Merge: Merges any overlapping entries (IoU >= 0.35) into a single master entry.
    3. Person-Bootstrap Fallback: If a person is stationary for >= 30 frames over a spot without
       a registered chair, synthesizes a bootstrap chair entry from lower-body estimation.
       When YOLO later detects a real chair at this spot, refines it with high-accuracy YOLO bbox.
    """

    def __init__(self, min_confidence=0.45, match_iou_threshold=0.35, bootstrap_persistence=30):
        self.min_confidence = min_confidence
        self.match_iou_threshold = match_iou_threshold
        self.bootstrap_persistence = bootstrap_persistence
        self.next_chair_id = 1

        # registry: chair_id -> dict {
        #    "id": int, "bbox": [x1,y1,x2,y2], "name": str, "conf": float,
        #    "detection_count": int, "is_bootstrap": bool
        # }
        self.registry = {}

        # Tracking stationary persons for bootstrap fallback:
        # person_id -> {"centroid": (cx, cy), "frames": int, "last_bbox": [x1,y1,x2,y2]}
        self.person_stability = {}

    def update(self, live_chair_detections, tracked_persons=None):
        """
        Processes live chair detections and tracked persons to update registry.
        """
        # Step 1: Process live YOLO chair detections with strict deduplication
        for det in live_chair_detections:
            conf = det.get("confidence", 1.0)
            if conf < self.min_confidence:
                continue

            bbox = det["bbox"]
            matched_id = None
            best_iou = 0.0

            # Check IoU against all existing registry entries (Deduplication)
            for cid, entry in self.registry.items():
                iou = compute_iou(bbox, entry["bbox"])
                if iou > best_iou and iou >= self.match_iou_threshold:
                    best_iou = iou
                    matched_id = cid

            if matched_id is not None:
                # Update existing entry position (Smooth Exponential Moving Average)
                old_bbox = self.registry[matched_id]["bbox"]
                alpha = 0.25
                smoothed = [
                    int(alpha * bbox[i] + (1 - alpha) * old_bbox[i]) for i in range(4)
                ]
                self.registry[matched_id]["bbox"] = smoothed
                self.registry[matched_id]["conf"] = max(conf, self.registry[matched_id]["conf"])
                self.registry[matched_id]["detection_count"] += 1

                # If previously a bootstrap entry, replace with high-accuracy YOLO bbox
                if self.registry[matched_id].get("is_bootstrap", False):
                    self.registry[matched_id]["is_bootstrap"] = False
                    print(f"[CHAIR REGISTRY] Refined Bootstrap Chair #{matched_id} with Real YOLO BBox {bbox}")

            else:
                # Add new distinct permanent chair entry
                cid = self.next_chair_id
                self.next_chair_id += 1
                self.registry[cid] = {
                    "id": cid,
                    "bbox": bbox,
                    "name": f"Chair {cid}",
                    "conf": conf,
                    "detection_count": 1,
                    "is_bootstrap": False
                }
                print(f"[CHAIR REGISTRY] Registered New Chair #{cid} at BBox {bbox} (conf={conf:.2f})")

        # Step 2: Fallback Bootstrap — Register chair from stationary persons if chair never visible empty
        if tracked_persons:
            self._bootstrap_from_persons(tracked_persons)

        # Step 3: Auto Merge duplicate overlapping entries in registry
        self.merge_duplicates()

        return self.registry

    def _bootstrap_from_persons(self, tracked_persons):
        """
        Synthesizes chair entry if a person remains stationary for >= bootstrap_persistence frames
        and does not overlap with any existing chair in registry.
        """
        active_pids = set(tracked_persons.keys())

        for pid, person in tracked_persons.items():
            full_bbox = person["bbox"]
            centroid = compute_centroid(full_bbox)

            if pid not in self.person_stability:
                self.person_stability[pid] = {
                    "centroid": centroid,
                    "frames": 1,
                    "last_bbox": full_bbox
                }
            else:
                prev_c = self.person_stability[pid]["centroid"]
                dist = math.hypot(centroid[0] - prev_c[0], centroid[1] - prev_c[1])

                if dist < 30.0:  # Shift < 30px is considered stationary
                    self.person_stability[pid]["frames"] += 1
                    self.person_stability[pid]["last_bbox"] = full_bbox
                else:
                    self.person_stability[pid] = {
                        "centroid": centroid,
                        "frames": 1,
                        "last_bbox": full_bbox
                    }

            # Check if stability threshold reached
            if self.person_stability[pid]["frames"] >= self.bootstrap_persistence:
                p_bbox = self.person_stability[pid]["last_bbox"]

                # Check if this person overlaps with ANY existing chair in registry (IoU >= 0.20)
                has_overlap = False
                for cid, entry in self.registry.items():
                    if compute_iou(p_bbox, entry["bbox"]) >= 0.20:
                        has_overlap = True
                        break

                if not has_overlap:
                    # Estimate chair bbox from person lower body
                    px1, py1, px2, py2 = p_bbox
                    pw = max(1, px2 - px1)
                    ph = max(1, py2 - py1)

                    seat_y1 = py1 + int(ph * 0.45)
                    seat_y2 = py2
                    pad_x = int(pw * 0.15)
                    est_chair_bbox = [max(0, px1 - pad_x), seat_y1, px2 + pad_x, seat_y2]

                    cid = self.next_chair_id
                    self.next_chair_id += 1

                    self.registry[cid] = {
                        "id": cid,
                        "bbox": est_chair_bbox,
                        "name": f"Chair {cid}",
                        "conf": 0.50,
                        "detection_count": 1,
                        "is_bootstrap": True
                    }
                    print(f"[CHAIR BOOTSTRAP] Person ID:{pid} stationary for {self.bootstrap_persistence}f -> Created Chair #{cid} at {est_chair_bbox}")

        # Clean up stale person tracking entries
        stale_pids = [p for p in self.person_stability if p not in active_pids]
        for p in stale_pids:
            del self.person_stability[p]

    def merge_duplicates(self):
        """
        Merges any duplicate entries in the registry that overlap each other (IoU >= 0.35).
        Retains entry with higher detection count / confidence.
        """
        cids = list(self.registry.keys())
        to_delete = set()

        for i in range(len(cids)):
            cid1 = cids[i]
            if cid1 in to_delete:
                continue

            for j in range(i + 1, len(cids)):
                cid2 = cids[j]
                if cid2 in to_delete:
                    continue

                entry1 = self.registry[cid1]
                entry2 = self.registry[cid2]

                iou = compute_iou(entry1["bbox"], entry2["bbox"])
                if iou >= self.match_iou_threshold:
                    # Duplicate found — pick winner based on detection count and non-bootstrap status
                    if entry1.get("is_bootstrap") and not entry2.get("is_bootstrap"):
                        to_delete.add(cid1)
                    elif not entry1.get("is_bootstrap") and entry2.get("is_bootstrap"):
                        to_delete.add(cid2)
                    elif entry1["detection_count"] >= entry2["detection_count"]:
                        to_delete.add(cid2)
                    else:
                        to_delete.add(cid1)

        for cid in to_delete:
            if cid in self.registry:
                print(f"[CHAIR REGISTRY MERGE] Merged duplicate Chair #{cid}")
                del self.registry[cid]

    def get_all_chairs(self):
        return self.registry
