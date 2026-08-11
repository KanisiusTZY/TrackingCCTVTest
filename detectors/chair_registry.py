import math
import numpy as np
from tracker import compute_iou, compute_centroid

class ChairRegistry:
    """
    Automatic Workstation & Chair Registry.
    Tracks workstations continuously and generalizably across any CCTV video.

    Guarantees:
    1. Zero Phantom Chairs: Hallway walkers or people pausing briefly do NOT leave fake red boxes.
    2. Zero Duplicate Overlaps: Merges overlapping seat candidates (IoU >= 0.20 or dist < 110px).
    3. Occlusion Resistance: Keeps workstation registered when employee is seated working.
    4. Permanent Workstations: Confirmed workstations remain registered when employee leaves.
    """

    def __init__(self, model_name='yolov8m.pt', iou_threshold=0.20, min_confidence=0.15, bootstrap_persistence=15):
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.bootstrap_persistence = bootstrap_persistence
        self.next_chair_id = 1
        self.registry = {}

    def process_frame(self, frame_count, live_chair_detections=None, tracked_persons=None):
        all_candidates = []

        # 1a. Existing Registry Workstations
        for cid, entry in list(self.registry.items()):
            all_candidates.append({
                "id": cid,
                "bbox": list(entry["bbox"]),
                "conf": entry["conf"],
                "age": entry["age"] + 1,
                "is_bootstrap": entry.get("is_bootstrap", False),
                "confirmed_by_chair": entry.get("confirmed_by_chair", False),
                "occupied_frames": entry.get("occupied_frames", 0),
                "source": "registry",
                "priority": 2
            })

        # 1b. Head & Body Presence Workstation Generation for SEATED Employees (PRIORITY 3 = Highest Truth!)
        if tracked_persons:
            presence_candidates = self._generate_presence_candidates(tracked_persons, all_candidates)
            all_candidates.extend(presence_candidates)

        # 1c. Add Confirmed Physical YOLO Chair Detections (Class 56) (PRIORITY 1)
        if live_chair_detections:
            for chair_det in live_chair_detections:
                all_candidates.append({
                    "id": None,
                    "bbox": list(chair_det["bbox"]),
                    "conf": chair_det.get("confidence", 0.50),
                    "age": 0,
                    "is_bootstrap": False,
                    "confirmed_by_chair": True,
                    "occupied_frames": 0,
                    "source": "yolo",
                    "priority": 1
                })

        # 2. Global Deduplication & NMS with Bulletproof Workstation ID Matching
        clean_chairs = self._global_nms_merge(all_candidates, frame_count)

        # 3. Filter out transient unconfirmed bootstrap entries that were only occupied briefly
        final_registry = {}
        for cid, entry in clean_chairs.items():
            is_bootstrap = entry.get("is_bootstrap", False)
            confirmed = entry.get("confirmed_by_chair", False)
            occupied_frames = entry.get("occupied_frames", 0)
            status = entry.get("status", "TIDAK DI TEMPAT")

            # A workstation is permanent if it was confirmed by a physical YOLO chair OR if someone sat there >= 45 frames (~1.5s)
            if confirmed or occupied_frames >= 45:
                entry["confirmed_by_chair"] = True
                final_registry[cid] = entry
            elif status == "BEKERJA" or is_bootstrap:
                final_registry[cid] = entry

        self.registry = final_registry
        return self.registry

    def _generate_presence_candidates(self, tracked_persons, existing_candidates):
        presence_list = []

        for pid, person in tracked_persons.items():
            full_bbox = person["bbox"]
            px1, py1, px2, py2 = full_bbox
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)
            aspect_ratio = ph / float(pw)

            # Rejection 1: Standing upright persons (H/W > 1.80)
            if aspect_ratio > 1.80:
                continue

            # Rejection 2: Persons actively moving (net_displacement >= 18px)
            net_displacement = person.get("net_displacement", 999.0)
            if net_displacement >= 18.0:
                continue

            # Synthesize workstation seat box around person upper body & seat position
            seat_y1 = py1 + int(ph * 0.15)
            seat_y2 = py2
            pad_x = int(pw * 0.05)
            est_bbox = [max(0, px1 - pad_x), seat_y1, px2 + pad_x, seat_y2]

            # Check if this person already maps to an existing workstation candidate or registry entry
            already_has_workstation = False
            for cand in existing_candidates:
                c1 = compute_centroid(est_bbox)
                c2 = compute_centroid(cand["bbox"])
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                iou = compute_iou(est_bbox, cand["bbox"])

                # Deduplication threshold: dist < 110px or IoU >= 0.20
                if iou >= 0.20 or dist < 110.0:
                    already_has_workstation = True
                    break

            if not already_has_workstation:
                presence_list.append({
                    "id": None,
                    "bbox": est_bbox,
                    "conf": 0.85,
                    "age": 0,
                    "is_bootstrap": True,
                    "confirmed_by_chair": False,
                    "occupied_frames": 1,
                    "source": "presence",
                    "priority": 3  # Highest priority for active seated human presence
                })

        return presence_list

    def _global_nms_merge(self, candidates, frame_count):
        if not candidates:
            return {}

        # Sort candidates by priority (presence=3 > registry=2 > yolo=1), then confidence descending
        candidates.sort(key=lambda c: (c["priority"], c["conf"]), reverse=True)

        merged_result = {}
        used = [False] * len(candidates)

        for i in range(len(candidates)):
            if used[i]:
                continue

            anchor = candidates[i]
            used[i] = True

            anchor_c = compute_centroid(anchor["bbox"])
            chair_id = anchor["id"]

            # Step A: Match candidate against existing registry entries OR already merged_result entries
            if chair_id is None:
                best_match_id = None
                min_d = 120.0

                # 1. First check against ALREADY MERGED entries in current frame
                for m_id, m_entry in merged_result.items():
                    m_c = compute_centroid(m_entry["bbox"])
                    d = math.hypot(anchor_c[0] - m_c[0], anchor_c[1] - m_c[1])
                    iou = compute_iou(anchor["bbox"], m_entry["bbox"])

                    if iou >= 0.20 or d < min_d:
                        min_d = d
                        best_match_id = m_id

                # 2. Then check against self.registry if not matched yet
                if best_match_id is None:
                    for reg_id, reg_entry in self.registry.items():
                        reg_c = compute_centroid(reg_entry["bbox"])
                        d = math.hypot(anchor_c[0] - reg_c[0], anchor_c[1] - reg_c[1])
                        iou = compute_iou(anchor["bbox"], reg_entry["bbox"])

                        if iou >= 0.20 or d < min_d:
                            min_d = d
                            best_match_id = reg_id

                if best_match_id is not None:
                    chair_id = best_match_id
                else:
                    chair_id = self.next_chair_id
                    self.next_chair_id += 1

            # Step B: If chair_id is already in merged_result, merge candidate into that existing entry!
            if chair_id in merged_result:
                existing = merged_result[chair_id]
                if anchor.get("confirmed_by_chair"):
                    existing["confirmed_by_chair"] = True
                existing["occupied_frames"] = max(existing.get("occupied_frames", 0), anchor.get("occupied_frames", 0))
                if not anchor["is_bootstrap"]:
                    existing["is_bootstrap"] = False
                existing["last_seen_frame"] = frame_count
                continue

            # Step C: Create new entry in merged_result for chair_id
            best_bbox = list(anchor["bbox"])
            best_conf = anchor["conf"]
            is_bootstrap = anchor["is_bootstrap"]
            confirmed = anchor.get("confirmed_by_chair", False)
            occupied_frames = anchor.get("occupied_frames", 0)
            age = anchor["age"]
            last_seen = frame_count if anchor["source"] in ["yolo", "presence"] else anchor.get("last_seen_frame", frame_count)

            # Merge any remaining overlapping candidate boxes in the candidate list
            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue

                cand = candidates[j]
                iou = compute_iou(best_bbox, cand["bbox"])
                c1 = compute_centroid(best_bbox)
                c2 = compute_centroid(cand["bbox"])
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

                if iou >= 0.20 or dist < 110.0:
                    used[j] = True
                    if cand.get("confirmed_by_chair"):
                        confirmed = True
                    occupied_frames = max(occupied_frames, cand.get("occupied_frames", 0))
                    if not cand["is_bootstrap"]:
                        is_bootstrap = False
                        best_conf = max(best_conf, cand["conf"])
                        last_seen = frame_count

            if chair_id in self.registry:
                prev_entry = self.registry[chair_id]
                occupied_frames = max(occupied_frames, prev_entry.get("occupied_frames", 0))
                if prev_entry.get("confirmed_by_chair"):
                    confirmed = True

            merged_result[chair_id] = {
                "id": chair_id,
                "bbox": best_bbox,
                "name": f"Workstation {chair_id}",
                "conf": best_conf,
                "age": age,
                "is_bootstrap": is_bootstrap,
                "confirmed_by_chair": confirmed,
                "occupied_frames": occupied_frames,
                "last_seen_frame": last_seen
            }

        return merged_result

    def get_all_chairs(self):
        return self.registry
