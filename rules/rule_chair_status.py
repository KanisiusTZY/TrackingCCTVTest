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
    Evaluates status PER DETECTED CHAIR.
    Matches upper_body_bbox of tracked persons against detected chair_bbox using IoU.
    
    Status states per chair:
    - "BEKERJA": upper_body_bbox overlap IoU >= threshold_occupied for persistence_frames.
    - "TIDAK DI TEMPAT": IoU < threshold_occupied for persistence_frames. Accumulates away_timer.
    """
    def __init__(self, enabled=True):
        super().__init__(name="Dynamic Chair Status (BEKERJA / TIDAK DI TEMPAT)", rule_id="rule_chair_status", enabled=enabled)
        self.occupied_counters = {}  # chair_id -> int
        self.empty_counters = {}     # chair_id -> int
        self.away_timers = {}        # chair_id -> float (seconds)
        self.prev_status = {}        # chair_id -> str ("BEKERJA" / "TIDAK DI TEMPAT")

    def process(self, tracked_objects, config, dt):
        """
        tracked_objects: dict {"persons": {p_id: person_data}, "chairs": {c_id: chair_data}}
        """
        if not self.enabled:
            return

        iou_thresh = config["thresholds"].get("iou_chair_occupied", 0.15)
        persistence = config["thresholds"].get("persistence_frames", 15)

        tracked_persons = tracked_objects.get("persons", {})
        tracked_chairs = tracked_objects.get("chairs", {})

        for chair_id, chair in tracked_chairs.items():
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

            # Counter update logic
            if max_iou >= iou_thresh:
                self.occupied_counters[chair_id] += 1
                self.empty_counters[chair_id] = 0
            else:
                self.empty_counters[chair_id] += 1
                self.occupied_counters[chair_id] = 0

            # Persistence gating + hysteresis status evaluation
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

            # Log status change to console
            if new_status != self.prev_status[chair_id]:
                if new_status == "TIDAK DI TEMPAT":
                    print(f"[CHAIR:{chair_id}] status changed: BEKERJA -> TIDAK DI TEMPAT (timer started)")
                else:
                    away_dur = format_duration(self.away_timers.get(chair_id, 0.0))
                    print(f"[CHAIR:{chair_id}] status changed: TIDAK DI TEMPAT -> BEKERJA (away duration was {away_dur})")

            self.prev_status[chair_id] = new_status

            # Attach status properties to chair object for rendering
            chair["status"] = new_status
            chair["away_timer"] = self.away_timers[chair_id]
            chair["away_label"] = f"TIDAK DI TEMPAT: {format_duration(self.away_timers[chair_id])}"

            if new_status == "BEKERJA" and best_person is not None:
                chair["matched_person_id"] = best_person["id"]
                chair["matched_upper_body_bbox"] = best_person.get("upper_body_bbox", best_person["bbox"])
            else:
                chair["matched_person_id"] = None
                chair["matched_upper_body_bbox"] = None

        # Clean up stale chair IDs no longer tracked
        stale_chair_ids = [cid for cid in self.prev_status if cid not in tracked_chairs]
        for cid in stale_chair_ids:
            del self.occupied_counters[cid]
            del self.empty_counters[cid]
            del self.away_timers[cid]
            del self.prev_status[cid]
