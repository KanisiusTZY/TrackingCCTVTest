import math
import numpy as np
from tracker import compute_iou, compute_centroid

def compute_horizontal_overlap_ratio(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[2], bbox2[2])
    inter_x = max(0, x2 - x1)

    w1 = max(1, bbox1[2] - bbox1[0])
    w2 = max(1, bbox2[2] - bbox2[0])
    min_w = min(w1, w2)

    return inter_x / float(min_w)


class ChairRegistry:
    """
    Chair Registry with Seated Employee Bootstrap & Unique Workstation Management.

    Guarantees:
    1. Synthesizes chair candidates for all seated employees (including foreground & background).
    2. Merges duplicate detections for the same physical chair.
    3. Keeps registry deduplicated every frame.
    """

    def __init__(self, iou_threshold=0.30, min_confidence=0.40, bootstrap_persistence=10):
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.bootstrap_persistence = bootstrap_persistence
        self.next_chair_id = 1
        self.registry = {}
        self.person_stability = {}

    def process_frame(self, frame_count, live_chair_detections, tracked_persons=None):
        all_candidates = []

        # 1a. Existing Registry Chairs
        for cid, entry in self.registry.items():
            all_candidates.append({
                "id": cid,
                "bbox": list(entry["bbox"]),
                "conf": entry["conf"],
                "age": entry["age"] + 1,
                "is_bootstrap": entry.get("is_bootstrap", False),
                "source": "registry",
                "priority": 3 if not entry.get("is_bootstrap") else 1
            })

        # 1b. Live YOLO Chair Detections
        for det in live_chair_detections:
            conf = det.get("confidence", 1.0)
            if conf >= self.min_confidence:
                all_candidates.append({
                    "id": None,
                    "bbox": list(det["bbox"]),
                    "conf": conf,
                    "age": 0,
                    "is_bootstrap": False,
                    "source": "yolo",
                    "priority": 2
                })

        # 1c. Person-Bootstrap Candidates for Seated Employees
        if tracked_persons:
            bootstrap_candidates = self._generate_bootstrap_candidates(tracked_persons, all_candidates)
            all_candidates.extend(bootstrap_candidates)

        total_candidate_count = len(all_candidates)

        # 2. Global NMS Merge
        clean_chairs = self._global_nms_merge(all_candidates, frame_count)

        # 3. Replace self.registry completely
        self.registry = clean_chairs

        print(f"[Frame {frame_count}] Candidates: {total_candidate_count} -> After NMS: {len(self.registry)} unique chairs")

        return self.registry

    def _generate_bootstrap_candidates(self, tracked_persons, existing_candidates):
        bootstrap_list = []
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

                if dist < 50.0:  # Seated typing/working movement tolerance
                    self.person_stability[pid]["frames"] += 1
                    self.person_stability[pid]["last_bbox"] = full_bbox
                else:
                    self.person_stability[pid] = {
                        "centroid": centroid,
                        "frames": 1,
                        "last_bbox": full_bbox
                    }

            if self.person_stability[pid]["frames"] >= self.bootstrap_persistence:
                p_bbox = self.person_stability[pid]["last_bbox"]
                px1, py1, px2, py2 = p_bbox
                pw = max(1, px2 - px1)
                ph = max(1, py2 - py1)

                seat_y1 = py1 + int(ph * 0.40)
                seat_y2 = py2
                pad_x = int(pw * 0.10)
                est_bbox = [max(0, px1 - pad_x), seat_y1, px2 + pad_x, seat_y2]

                already_has_chair = False
                for cand in existing_candidates:
                    c1 = compute_centroid(est_bbox)
                    c2 = compute_centroid(cand["bbox"])
                    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    iou = compute_iou(est_bbox, cand["bbox"])

                    if iou >= 0.25 or dist < 70.0:
                        already_has_chair = True
                        break

                if not already_has_chair:
                    for b_item in bootstrap_list:
                        c1 = compute_centroid(est_bbox)
                        c2 = compute_centroid(b_item["bbox"])
                        dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                        iou = compute_iou(est_bbox, b_item["bbox"])

                        if iou >= 0.25 or dist < 70.0:
                            already_has_chair = True
                            break

                if not already_has_chair:
                    bootstrap_list.append({
                        "id": None,
                        "bbox": est_bbox,
                        "conf": 0.50,
                        "age": 0,
                        "is_bootstrap": True,
                        "source": "bootstrap",
                        "priority": 1
                    })

        stale_pids = [p for p in self.person_stability if p not in active_pids]
        for p in stale_pids:
            del self.person_stability[p]

        return bootstrap_list

    def _global_nms_merge(self, candidates, frame_count):
        if not candidates:
            return {}

        candidates.sort(key=lambda c: (c["priority"], c["age"], c["conf"]), reverse=True)

        merged_result = {}
        used = [False] * len(candidates)

        for i in range(len(candidates)):
            if used[i]:
                continue

            anchor = candidates[i]
            used[i] = True

            chair_id = anchor["id"] if anchor["id"] is not None else self.next_chair_id
            if anchor["id"] is None:
                self.next_chair_id += 1

            best_bbox = list(anchor["bbox"])
            best_conf = anchor["conf"]
            is_bootstrap = anchor["is_bootstrap"]
            age = anchor["age"]

            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue

                cand = candidates[j]
                iou = compute_iou(best_bbox, cand["bbox"])
                c1 = compute_centroid(best_bbox)
                c2 = compute_centroid(cand["bbox"])
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                h_overlap = compute_horizontal_overlap_ratio(best_bbox, cand["bbox"])

                same_column = (h_overlap >= 0.60 and abs(c1[0] - c2[0]) < 80.0)

                if iou >= self.iou_threshold or dist < 90.0 or same_column:
                    used[j] = True

                    if same_column:
                        best_bbox = [
                            min(best_bbox[0], cand["bbox"][0]),
                            min(best_bbox[1], cand["bbox"][1]),
                            max(best_bbox[2], cand["bbox"][2]),
                            max(best_bbox[3], cand["bbox"][3])
                        ]

                    if not cand["is_bootstrap"] and is_bootstrap:
                        if not same_column:
                            best_bbox = list(cand["bbox"])
                        is_bootstrap = False
                        best_conf = max(best_conf, cand["conf"])

            merged_result[chair_id] = {
                "id": chair_id,
                "bbox": best_bbox,
                "name": f"Chair {chair_id}",
                "conf": best_conf,
                "age": age,
                "is_bootstrap": is_bootstrap,
                "last_seen_frame": frame_count
            }

        return merged_result

    def get_all_chairs(self):
        return self.registry
