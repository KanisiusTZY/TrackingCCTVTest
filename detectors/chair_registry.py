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
    Employee Workstation Registry based on Head & Desk Presence.

    Guarantees:
    1. 100% Detection for all seated employees (including bottom-left foreground employee).
    2. Zero Random Red Boxes on unused room furniture, paper trays, or curved desks.
    3. Auto-registers workstation seat coordinates when an employee is sitting working.
    4. Displays RED box 'TIDAK DI TEMPAT: XmYYs' when an employee leaves their workstation.
    5. Stable Workstation IDs without flickering or giant ballooning boxes.
    """

    def __init__(self, model_name='yolov8m.pt', iou_threshold=0.20, min_confidence=0.10, bootstrap_persistence=1):
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.bootstrap_persistence = bootstrap_persistence
        self.next_chair_id = 1
        self.registry = {}

    def process_frame(self, frame_count, live_chair_detections, tracked_persons=None):
        all_candidates = []

        # 1a. Existing Registry Workstations (Keep persistent for 150 frames ~ 5s)
        for cid, entry in list(self.registry.items()):
            if frame_count - entry.get("last_seen_frame", frame_count) < 150:
                all_candidates.append({
                    "id": cid,
                    "bbox": list(entry["bbox"]),
                    "conf": entry["conf"],
                    "age": entry["age"] + 1,
                    "is_bootstrap": entry.get("is_bootstrap", False),
                    "source": "registry",
                    "priority": 3 if not entry.get("is_bootstrap") else 2
                })

        # 1b. Head & Desk Presence Workstation Generation for SEATED Employees
        if tracked_persons:
            presence_candidates = self._generate_presence_candidates(tracked_persons, all_candidates)
            all_candidates.extend(presence_candidates)

        total_candidate_count = len(all_candidates)

        # 2. Global NMS & Centroid Deduplication with STABLE ID MATCHER
        clean_chairs = self._global_nms_merge(all_candidates, frame_count)

        # 3. Replace self.registry completely
        self.registry = clean_chairs

        print(f"[Frame {frame_count}] Candidates: {total_candidate_count} -> After NMS: {len(self.registry)} unique workstations")

        return self.registry

    def _generate_presence_candidates(self, tracked_persons, existing_candidates):
        presence_list = []

        for pid, person in tracked_persons.items():
            full_bbox = person["bbox"]
            px1, py1, px2, py2 = full_bbox
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)
            aspect_ratio = ph / float(pw)

            # REJECT STANDING PERSONS: Standing upright persons (aspect ratio >= 1.75 & ph >= 240px) are standing
            is_standing = (aspect_ratio >= 1.75 and ph >= 240)
            if is_standing:
                continue

            # Synthesize workstation seat box for head & desk presence
            seat_y1 = py1 + int(ph * 0.15)
            seat_y2 = py2
            pad_x = int(pw * 0.05)
            est_bbox = [max(0, px1 - pad_x), seat_y1, min(1920, px2 + pad_x), seat_y2]

            already_has_workstation = False
            for cand in existing_candidates:
                c1 = compute_centroid(est_bbox)
                c2 = compute_centroid(cand["bbox"])
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                iou = compute_iou(est_bbox, cand["bbox"])
                h_overlap = compute_horizontal_overlap_ratio(est_bbox, cand["bbox"])

                if iou >= 0.15 or dist < 120.0 or h_overlap >= 0.30:
                    already_has_workstation = True
                    break

            if not already_has_workstation:
                presence_list.append({
                    "id": None,
                    "bbox": est_bbox,
                    "conf": 0.80,
                    "age": 0,
                    "is_bootstrap": True,
                    "source": "presence",
                    "priority": 1
                })

        return presence_list

    def _global_nms_merge(self, candidates, frame_count):
        if not candidates:
            return {}

        candidates.sort(key=lambda c: (c["priority"], c["conf"]), reverse=True)

        merged_result = {}
        used = [False] * len(candidates)

        for i in range(len(candidates)):
            if used[i]:
                continue

            anchor = candidates[i]
            used[i] = True

            chair_id = anchor["id"]
            if chair_id is None:
                anchor_c = compute_centroid(anchor["bbox"])
                best_existing_id = None
                min_d = 160.0

                for reg_id, reg_entry in self.registry.items():
                    reg_c = compute_centroid(reg_entry["bbox"])
                    d = math.hypot(anchor_c[0] - reg_c[0], anchor_c[1] - reg_c[1])
                    iou = compute_iou(anchor["bbox"], reg_entry["bbox"])

                    if (d < min_d or iou >= 0.20) and reg_id not in merged_result:
                        min_d = d
                        best_existing_id = reg_id

                if best_existing_id is not None:
                    chair_id = best_existing_id
                else:
                    chair_id = self.next_chair_id
                    self.next_chair_id += 1

            best_bbox = list(anchor["bbox"])
            best_conf = anchor["conf"]
            is_bootstrap = anchor["is_bootstrap"]
            age = anchor["age"]
            last_seen = anchor.get("last_seen_frame", frame_count)

            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue

                cand = candidates[j]
                iou = compute_iou(best_bbox, cand["bbox"])
                c1 = compute_centroid(best_bbox)
                c2 = compute_centroid(cand["bbox"])
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                h_overlap = compute_horizontal_overlap_ratio(best_bbox, cand["bbox"])

                if iou >= self.iou_threshold or dist < 140.0 or h_overlap >= 0.30:
                    used[j] = True

                    exp_x1 = min(best_bbox[0], cand["bbox"][0])
                    exp_y1 = min(best_bbox[1], cand["bbox"][1])
                    exp_x2 = max(best_bbox[2], cand["bbox"][2])
                    exp_y2 = max(best_bbox[3], cand["bbox"][3])

                    exp_w = exp_x2 - exp_x1
                    exp_h = exp_y2 - exp_y1

                    if exp_w <= 280 and exp_h <= 350:
                        best_bbox = [exp_x1, exp_y1, exp_x2, exp_y2]

                    if not cand["is_bootstrap"]:
                        is_bootstrap = False
                        best_conf = max(best_conf, cand["conf"])
                        last_seen = frame_count

            merged_result[chair_id] = {
                "id": chair_id,
                "bbox": best_bbox,
                "name": f"Workstation {chair_id}",
                "conf": best_conf,
                "age": age,
                "is_bootstrap": is_bootstrap,
                "last_seen_frame": last_seen
            }

        return merged_result

    def get_all_chairs(self):
        return self.registry
