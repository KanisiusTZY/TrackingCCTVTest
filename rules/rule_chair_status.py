from rules.base_rule import BaseRule
from tracker import compute_iou

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
    TOTAL REWRITE: Evaluates status per UNIQUE chair from clean_chairs (ChairRegistry output).

    Guarantees:
    - EXACTLY ONE status per chair_id per frame ("BEKERJA" or "TIDAK DI TEMPAT").
    - "BEKERJA": Green box around person upper_body_bbox.
    - "TIDAK DI TEMPAT": Red box around clean chair bbox + duration timer.
    """
    def __init__(self, enabled=True):
        super().__init__(name="Dynamic Chair Status (BEKERJA / TIDAK DI TEMPAT)", rule_id="rule_chair_status", enabled=enabled)
        self.occupied_counters = {}  # chair_id -> int
        self.empty_counters = {}     # chair_id -> int
        self.away_timers = {}        # chair_id -> float
        self.prev_status = {}        # chair_id -> str

    def process(self, tracked_persons, clean_chairs, config, dt):
        """
        tracked_persons: dict {p_id: person_data}
        clean_chairs: dict {c_id: chair_data} from ChairRegistry
        """
        if not self.enabled:
            return

        iou_thresh = config["thresholds"].get("iou_chair_occupied", 0.15)
        persistence = config["thresholds"].get("persistence_frames", 15)

        for chair_id, chair in clean_chairs.items():
            chair_bbox = chair["bbox"]

            if chair_id not in self.occupied_counters:
                self.occupied_counters[chair_id] = 0
                self.empty_counters[chair_id] = 0
                self.away_timers[chair_id] = 0.0
                self.prev_status[chair_id] = "TIDAK DI TEMPAT"

            # Find upper_body_bbox of person with highest IoU against chair_bbox
            max_iou = 0.0
            best_person = None

            for person_id, person in tracked_persons.items():
                upper_bbox = person.get("upper_body_bbox", person["bbox"])
                iou = compute_iou(chair_bbox, upper_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_person = person

            # IoU Threshold evaluation
            if max_iou >= iou_thresh:
                self.occupied_counters[chair_id] += 1
                self.empty_counters[chair_id] = 0
            else:
                self.empty_counters[chair_id] += 1
                self.occupied_counters[chair_id] = 0

            # Persistence Gating & Hysteresis
            if self.occupied_counters[chair_id] >= persistence:
                new_status = "BEKERJA"
                self.away_timers[chair_id] = 0.0
            elif self.empty_counters[chair_id] >= persistence:
                new_status = "TIDAK DI TEMPAT"
                self.away_timers[chair_id] += dt
            else:
                # Keep previous status (hysteresis)
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

            # Attach properties to chair object for visualizer rendering
            chair["status"] = new_status
            chair["away_timer"] = self.away_timers[chair_id]
            chair["away_label"] = f"TIDAK DI TEMPAT: {format_duration(self.away_timers[chair_id])}"

            if new_status == "BEKERJA" and best_person is not None:
                chair["matched_person_id"] = best_person["id"]
                chair["matched_upper_body_bbox"] = best_person.get("upper_body_bbox", best_person["bbox"])
            else:
                chair["matched_person_id"] = None
                chair["matched_upper_body_bbox"] = None

        # Clean up stale chair entries from tracking counters
        stale_cids = [cid for cid in self.prev_status if cid not in clean_chairs]
        for cid in stale_cids:
            del self.occupied_counters[cid]
            del self.empty_counters[cid]
            del self.away_timers[cid]
            del self.prev_status[cid]
