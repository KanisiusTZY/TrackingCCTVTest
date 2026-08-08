from rules.base_rule import BaseRule
from tracker import compute_iou, compute_centroid
import math

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
    Evaluates status per UNIQUE chair from clean_chairs with Person-Chair Exclusivity.

    Guarantees:
    - Exactly ONE status per chair_id per frame.
    - Person-Chair Exclusivity: If a person is working at a seat (BEKERJA), any nearby
      unassigned chair bbox overlapping that person is suppressed, preventing stray red boxes!
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

        iou_thresh = config["thresholds"].get("iou_chair_occupied", 0.15)
        persistence = config["thresholds"].get("persistence_frames", 15)

        # Track which person IDs are already assigned to a BEKERJA chair
        assigned_person_ids = set()

        # Step 1: First pass — evaluate chairs against unassigned persons
        for chair_id, chair in clean_chairs.items():
            chair_bbox = chair["bbox"]

            if chair_id not in self.occupied_counters:
                self.occupied_counters[chair_id] = 0
                self.empty_counters[chair_id] = 0
                self.away_timers[chair_id] = 0.0
                self.prev_status[chair_id] = "TIDAK DI TEMPAT"

            max_iou = 0.0
            best_person = None

            for person_id, person in tracked_persons.items():
                upper_bbox = person.get("upper_body_bbox", person["bbox"])
                iou = compute_iou(chair_bbox, upper_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_person = person

            # Also check full body distance as fallback match
            if max_iou < iou_thresh and best_person is not None:
                p_c = compute_centroid(best_person["bbox"])
                c_c = compute_centroid(chair_bbox)
                dist = math.hypot(p_c[0] - c_c[0], p_c[1] - c_c[1])
                if dist < 100.0:
                    max_iou = iou_thresh

            if max_iou >= iou_thresh and best_person is not None:
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

        # Step 2: Second pass — Exclusivity suppression.
        # If any chair is marked "TIDAK DI TEMPAT", but it overlaps a person who is ALREADY BEKERJA elsewhere,
        # suppress the stray red box by hiding it or marking it suppressed.
        for chair_id, chair in clean_chairs.items():
            if chair["status"] == "TIDAK DI TEMPAT":
                for person_id, person in tracked_persons.items():
                    if person_id in assigned_person_ids:
                        p_bbox = person["bbox"]
                        iou = compute_iou(chair["bbox"], p_bbox)
                        p_c = compute_centroid(p_bbox)
                        c_c = compute_centroid(chair["bbox"])
                        dist = math.hypot(p_c[0] - c_c[0], p_c[1] - c_c[1])

                        if iou >= 0.15 or dist < 120.0:
                            # Suppress stray red box over an active workstation
                            chair["suppressed"] = True
                            break

        stale_cids = [cid for cid in self.prev_status if cid not in clean_chairs]
        for cid in stale_cids:
            del self.occupied_counters[cid]
            del self.empty_counters[cid]
            del self.away_timers[cid]
            del self.prev_status[cid]
