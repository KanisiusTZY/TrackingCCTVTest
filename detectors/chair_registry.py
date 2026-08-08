import math
import numpy as np
from tracker import compute_iou, compute_centroid

class ChairRegistry:
    """
    TOTAL REWRITE: Global Frame-by-Frame Chair Registry with Per-Frame NMS Cleanup.

    Guarantees:
    1. ZERO duplicate entries in the registry at any frame.
    2. Every frame performs a GLOBAL NMS / Merge across:
       - Existing persistent registry chairs
       - Live YOLO chair detections (conf >= 0.35)
       - Person-bootstrap chair candidates (stationary persons >= 30 frames)
    3. Replaces self.registry with the deduplicated clean list EVERY FRAME.
    4. Console debug logging: "[Frame X] Candidates: Y -> After NMS: Z unique chairs"
    """

    def __init__(self, iou_threshold=0.30, min_confidence=0.35, bootstrap_persistence=30):
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.bootstrap_persistence = bootstrap_persistence
        self.next_chair_id = 1

        # registry: dict of chair_id -> dict {
        #   "id": int, "bbox": [x1,y1,x2,y2], "conf": float,
        #   "age": int, "is_bootstrap": bool, "last_seen_frame": int
        # }
        self.registry = {}

        # Stationary person tracking for bootstrap fallback:
        # pid -> {"centroid": (cx,cy), "frames": int, "last_bbox": [x1,y1,x2,y2]}
        self.person_stability = {}

    def process_frame(self, frame_count, live_chair_detections, tracked_persons=None):
        """
        Executes the total per-frame cleanup pipeline.
        Returns clean_chairs dict {chair_id: chair_dict}.
        """
        all_candidates = []

        # -------------------------------------------------------------
        # 1a. Collect Existing Registry Chairs as Candidates
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 1b. Collect Live YOLO Chair Detections as Candidates
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # 1c. Collect Person-Bootstrap Candidates (Stationary Persons)
        # -------------------------------------------------------------
        if tracked_persons:
            bootstrap_candidates = self._generate_bootstrap_candidates(tracked_persons)
            all_candidates.extend(bootstrap_candidates)

        total_candidate_count = len(all_candidates)

        # -------------------------------------------------------------
        # 2. Global NMS / Merge Across ALL Candidates (IoU >= 0.30)
        # -------------------------------------------------------------
        clean_chairs = self._global_nms_merge(all_candidates, frame_count)

        # -------------------------------------------------------------
        # 3. Replace self.registry completely with clean_chairs
        # -------------------------------------------------------------
        self.registry = clean_chairs

        # Debug console output required by specification
        print(f"[Frame {frame_count}] Candidates: {total_candidate_count} -> After NMS: {len(self.registry)} unique chairs")

        return self.registry

    def _generate_bootstrap_candidates(self, tracked_persons):
        """
        Identifies stationary persons (shift < 30px for >= 30 frames)
        and returns estimated chair candidates.
        """
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

                if dist < 30.0:
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

                # Estimate lower body seat bbox
                seat_y1 = py1 + int(ph * 0.45)
                seat_y2 = py2
                pad_x = int(pw * 0.15)
                est_bbox = [max(0, px1 - pad_x), seat_y1, px2 + pad_x, seat_y2]

                bootstrap_list.append({
                    "id": None,
                    "bbox": est_bbox,
                    "conf": 0.50,
                    "age": 0,
                    "is_bootstrap": True,
                    "source": "bootstrap",
                    "priority": 1
                })

        # Cleanup stale person stability trackers
        stale_pids = [p for p in self.person_stability if p not in active_pids]
        for p in stale_pids:
            del self.person_stability[p]

        return bootstrap_list

    def _global_nms_merge(self, candidates, frame_count):
        """
        Global NMS & Merge logic:
        Sorts candidates by priority (Existing Registry > Real YOLO > Bootstrap) and Age.
        Merges candidates with IoU >= self.iou_threshold into a single unique chair.
        """
        if not candidates:
            return {}

        # Sort candidates: highest priority first, then highest age, then highest conf
        candidates.sort(key=lambda c: (c["priority"], c["age"], c["conf"]), reverse=True)

        merged_result = {}
        used = [False] * len(candidates)

        for i in range(len(candidates)):
            if used[i]:
                continue

            anchor = candidates[i]
            used[i] = True

            # Determine chair_id
            if anchor["id"] is not None:
                chair_id = anchor["id"]
            else:
                chair_id = self.next_chair_id
                self.next_chair_id += 1

            best_bbox = list(anchor["bbox"])
            best_conf = anchor["conf"]
            is_bootstrap = anchor["is_bootstrap"]
            age = anchor["age"]

            # Merge any overlapping candidate into this anchor
            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue

                cand = candidates[j]
                iou = compute_iou(best_bbox, cand["bbox"])

                if iou >= self.iou_threshold:
                    used[j] = True

                    # If candidate is a real YOLO detection and anchor is bootstrap, update bbox to YOLO
                    if not cand["is_bootstrap"] and is_bootstrap:
                        best_bbox = list(cand["bbox"])
                        is_bootstrap = False
                        best_conf = max(best_conf, cand["conf"])
                    elif not cand["is_bootstrap"] and not is_bootstrap:
                        # Smooth position using moving average
                        alpha = 0.30
                        best_bbox = [
                            int(alpha * cand["bbox"][k] + (1 - alpha) * best_bbox[k])
                            for k in range(4)
                        ]
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
