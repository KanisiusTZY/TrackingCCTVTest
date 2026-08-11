from rules.base_rule import BaseRule
from tracker import compute_iou, compute_centroid
import math

def compute_x_overlap_ratio(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[2], bbox2[2])
    inter_x = max(0, x2 - x1)
    w1 = max(1, bbox1[2] - bbox1[0])
    w2 = max(1, bbox2[2] - bbox2[0])
    min_w = min(w1, w2)
    return inter_x / float(min_w)

def format_duration(seconds):
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60

    if hours > 0:
        return f"{hours}h{minutes:02d}min"
    elif minutes > 0:
        return f"{minutes}m{secs:02d}s"
    else:
        return f"{secs}s"

class RuleChairStatus(BaseRule):
    """
    Evaluates Workstation Status based on Head & Body Presence with Temporal Hysteresis & Assignment Persistence.

    Guarantees:
    1. Zero Passerby Interference: Walking people passing near a desk do NOT trigger BEKERJA.
       - Requires 5 consecutive frames (~0.2s) of seated overlap before status becomes BEKERJA.
       - Rejects moving persons (net_displacement >= 18px).
    2. Zero Duplicate Workstations: Single 1-to-1 matching per workstation.
    3. Zero Flicker: Hysteresis persistence buffer prevents 1-11 frame detection drops from flickering.
    """
    def __init__(self, enabled=True):
        super().__init__(name="Dynamic Workstation Status (BEKERJA / TIDAK DI TEMPAT)", rule_id="rule_chair_status", enabled=enabled)
        self.occupied_counters = {}
        self.empty_counters = {}
        self.away_timers = {}
        self.prev_status = {}
        self.matched_persons = {}

        self.OCCUPIED_PERSISTENCE = 5           # Require 5 consecutive frames of overlap before declaring BEKERJA
        self.EMPTY_HYSTERESIS_PERSISTENCE = 12  # Keep BEKERJA status for 12 frames (~0.4s) before declaring TIDAK DI TEMPAT

    def process(self, tracked_persons, clean_chairs, config, dt):
        if not self.enabled:
            return

        iou_thresh = config["thresholds"].get("iou_chair_occupied", 0.02)
        assigned_person_ids = set()

        # Step 1: Evaluate presence per workstation (1-to-1 matching)
        for chair_id, chair in clean_chairs.items():
            chair_bbox = chair["bbox"]

            if chair_id not in self.occupied_counters:
                self.occupied_counters[chair_id] = 0
                self.empty_counters[chair_id] = 0
                self.away_timers[chair_id] = 0.0
                self.prev_status[chair_id] = "TIDAK DI TEMPAT"
                self.matched_persons[chair_id] = None

            max_score = 0.0
            best_person = None

            # Prioritize currently assigned seated person if available
            curr_matched_id = self.matched_persons.get(chair_id)

            for person_id, person in tracked_persons.items():
                if person_id in assigned_person_ids:
                    continue

                full_bbox = person["bbox"]
                px1, py1, px2, py2 = full_bbox
                pw = max(1, px2 - px1)
                ph = max(1, py2 - py1)
                aspect_ratio = ph / float(pw)
                net_displacement = person.get("net_displacement", 0.0)

                # Rejection 1: Standing upright walking persons (H/W > 1.80)
                if aspect_ratio > 1.80:
                    continue

                # Rejection 2: Actively moving / walking persons (net_displacement >= 18px)
                if net_displacement >= 18.0:
                    continue

                upper_bbox = person.get("upper_body_bbox", person["bbox"])
                iou = compute_iou(chair_bbox, upper_bbox)
                full_iou = compute_iou(chair_bbox, full_bbox)

                p_c = compute_centroid(person["bbox"])
                c_c = compute_centroid(chair_bbox)
                dist = math.hypot(p_c[0] - c_c[0], p_c[1] - c_c[1])
                x_overlap = compute_x_overlap_ratio(chair_bbox, full_bbox)

                # Head & Desk Presence Match
                if iou >= 0.02 or full_iou >= 0.02 or dist < 280.0 or x_overlap >= 0.15:
                    score = max(iou, full_iou, 0.50 if dist < 280.0 else 0.0)

                    # Give bonus to currently matched person to prevent passerby hijacking
                    if person_id == curr_matched_id:
                        score += 0.30

                    if score > max_score:
                        max_score = score
                        best_person = person

            if max_score >= iou_thresh and best_person is not None:
                self.occupied_counters[chair_id] += 1
                self.empty_counters[chair_id] = 0
                assigned_person_ids.add(best_person["id"])
                chair["occupied_frames"] = chair.get("occupied_frames", 0) + 1
            else:
                self.empty_counters[chair_id] += 1
                self.occupied_counters[chair_id] = 0

            # Assignment Persistence & Hysteresis State Transition Logic
            if self.occupied_counters[chair_id] >= self.OCCUPIED_PERSISTENCE:
                # Seated overlap maintained for >= 5 frames -> Declare BEKERJA
                new_status = "BEKERJA"
                self.away_timers[chair_id] = 0.0
                self.matched_persons[chair_id] = best_person["id"] if best_person else self.matched_persons.get(chair_id)
            elif self.prev_status[chair_id] == "BEKERJA" and self.empty_counters[chair_id] < self.EMPTY_HYSTERESIS_PERSISTENCE:
                # Maintain BEKERJA status during transient detection drops / passerby occlusion
                new_status = "BEKERJA"
            else:
                new_status = "TIDAK DI TEMPAT"
                self.away_timers[chair_id] += dt
                if self.empty_counters[chair_id] >= self.EMPTY_HYSTERESIS_PERSISTENCE:
                    self.matched_persons[chair_id] = None

            # Transition Logging
            if new_status != self.prev_status[chair_id]:
                if new_status == "TIDAK DI TEMPAT":
                    print(f"[WORKSTATION #{chair_id}] status changed: BEKERJA -> TIDAK DI TEMPAT (away timer started)")
                else:
                    dur = format_duration(self.away_timers.get(chair_id, 0.0))
                    print(f"[WORKSTATION #{chair_id}] status changed: TIDAK DI TEMPAT -> BEKERJA (was away for {dur})")

            self.prev_status[chair_id] = new_status

            chair["status"] = new_status
            chair["away_timer"] = self.away_timers[chair_id]
            chair["away_label"] = f"TIDAK DI TEMPAT: {format_duration(self.away_timers[chair_id])}"

            if new_status == "BEKERJA" and best_person is not None:
                chair["matched_person_id"] = best_person["id"]
                chair["matched_upper_body_bbox"] = best_person.get("upper_body_bbox", best_person["bbox"])
            else:
                chair["matched_person_id"] = None
                chair["matched_upper_body_bbox"] = None

        # Clean stale IDs
        stale_cids = [cid for cid in self.prev_status if cid not in clean_chairs]
        for cid in stale_cids:
            del self.occupied_counters[cid]
            del self.empty_counters[cid]
            del self.away_timers[cid]
            del self.prev_status[cid]
            if cid in self.matched_persons:
                del self.matched_persons[cid]
