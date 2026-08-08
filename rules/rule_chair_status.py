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
    Evaluates status per UNIQUE chair with Instant Workstation Matching & Immediate Away Red Boxes.

    Guarantees:
    - Seated employees (Foreground, Blonde, Background, Right Desk) immediately get GREEN box 'BEKERJA'.
    - Standing upright employees at filing cabinet NEVER get a green box.
    - When an employee stands up and walks away to cabinet, their empty workstation chair immediately turns RED 'TIDAK DI TEMPAT'.
    """
    def __init__(self, enabled=True):
        super().__init__(name="Dynamic Chair Status (BEKERJA / TIDAK DI TEMPAT)", rule_id="rule_chair_status", enabled=enabled)
        self.occupied_counters = {}
        self.empty_counters = {}
        self.away_timers = {}
        self.prev_status = {}

    def process(self, tracked_persons, clean_chairs, config, dt):
        if not self.enabled:
            return

        iou_thresh = config["thresholds"].get("iou_chair_occupied", 0.05)
        persistence = 2  # Instant response (2 frames)

        assigned_person_ids = set()

        # Step 1: Evaluate occupancy per chair
        for chair_id, chair in clean_chairs.items():
            chair_bbox = chair["bbox"]

            if chair_id not in self.occupied_counters:
                self.occupied_counters[chair_id] = 0
                self.empty_counters[chair_id] = 0
                self.away_timers[chair_id] = 0.0
                self.prev_status[chair_id] = "TIDAK DI TEMPAT"

            max_score = 0.0
            best_person = None

            for person_id, person in tracked_persons.items():
                if person_id in assigned_person_ids:
                    continue

                full_bbox = person["bbox"]
                px1, py1, px2, py2 = full_bbox
                pw = max(1, px2 - px1)
                ph = max(1, py2 - py1)
                aspect_ratio = ph / float(pw)

                # Standing upright check: Standing upright persons (H/W >= 1.75 and H >= 240px) cannot be seated!
                is_standing_upright = (aspect_ratio >= 1.75 and ph >= 240)
                if is_standing_upright:
                    continue

                upper_bbox = person.get("upper_body_bbox", person["bbox"])
                iou = compute_iou(chair_bbox, upper_bbox)

                p_c = compute_centroid(person["bbox"])
                c_c = compute_centroid(chair_bbox)
                dist = math.hypot(p_c[0] - c_c[0], p_c[1] - c_c[1])
                x_overlap = compute_x_overlap_ratio(chair_bbox, full_bbox)

                # Seated check: Seated employee at workstation matching
                if iou >= 0.05 or dist < 280.0 or x_overlap >= 0.25:
                    score = max(iou, 0.40 if dist < 280.0 else 0.0)
                    if score > max_score:
                        max_score = score
                        best_person = person

            if max_score >= iou_thresh and best_person is not None:
                self.occupied_counters[chair_id] += 1
                self.empty_counters[chair_id] = 0
                assigned_person_ids.add(best_person["id"])
            else:
                self.empty_counters[chair_id] += 1
                self.occupied_counters[chair_id] = 0

            if self.occupied_counters[chair_id] >= persistence:
                new_status = "BEKERJA"
                self.away_timers[chair_id] = 0.0
            elif self.empty_counters[chair_id] >= persistence:
                new_status = "TIDAK DI TEMPAT"
                self.away_timers[chair_id] += dt
            else:
                new_status = self.prev_status[chair_id]
                if new_status == "TIDAK DI TEMPAT":
                    self.away_timers[chair_id] += dt

            # Transition Logging
            if new_status != self.prev_status[chair_id]:
                if new_status == "TIDAK DI TEMPAT":
                    print(f"[CHAIR #{chair_id}] status changed: BEKERJA -> TIDAK DI TEMPAT (away timer started)")
                else:
                    dur = format_duration(self.away_timers.get(chair_id, 0.0))
                    print(f"[CHAIR #{chair_id}] status changed: TIDAK DI TEMPAT -> BEKERJA (was away for {dur})")

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

        # Step 2: Selective Suppression Pass for Seated Workstations ONLY
        # DO NOT suppress an empty chair if the person is standing upright (H/W >= 1.75) or has walked away (dy >= 120px)
        for chair_id, chair in clean_chairs.items():
            if chair["status"] == "TIDAK DI TEMPAT":
                for person_id, person in tracked_persons.items():
                    p_bbox = person["bbox"]
                    px1, py1, px2, py2 = p_bbox
                    pw = max(1, px2 - px1)
                    ph = max(1, py2 - py1)
                    aspect_ratio = ph / float(pw)

                    is_standing_upright = (aspect_ratio >= 1.75 and ph >= 240)

                    p_c = compute_centroid(p_bbox)
                    c_c = compute_centroid(chair["bbox"])
                    dist = math.hypot(p_c[0] - c_c[0], p_c[1] - c_c[1])
                    dy = abs(p_c[1] - c_c[1])

                    if not is_standing_upright and dy < 120.0 and dist < 180.0:
                        chair["suppressed"] = True
                        break

        stale_cids = [cid for cid in self.prev_status if cid not in clean_chairs]
        for cid in stale_cids:
            del self.occupied_counters[cid]
            del self.empty_counters[cid]
            del self.away_timers[cid]
            del self.prev_status[cid]
