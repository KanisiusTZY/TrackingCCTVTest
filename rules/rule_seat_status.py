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


class RuleSeatStatus(BaseRule):
    """
    Unified seat zone rule: each chair_zone is either BEKERJA (occupied, green)
    or TIDAK DI TEMPAT (empty, red + timer). Determined purely by IoU overlap
    between person bounding boxes and static chair zones — no pose analysis.
    """

    def __init__(self, enabled=True):
        super().__init__(
            name="Seat Status (Bekerja / Tidak di Tempat)",
            rule_id="rule_seat_status",
            enabled=enabled,
        )
        self.occupied_counters = {}   # chair_id -> consecutive occupied frame count
        self.empty_counters = {}      # chair_id -> consecutive empty frame count
        self.away_timers = {}         # chair_id -> accumulated seconds while empty
        self.prev_status = {}         # chair_id -> last confirmed status string

    def process(self, tracked_persons, chair_zones, config, dt):
        if not self.enabled:
            return

        iou_thresh = config["thresholds"].get("iou_chair_occupied", 0.20)
        persistence = config["thresholds"].get("persistence_frames", 15)

        for chair in chair_zones:
            cid = chair["id"]
            cbox = chair["bbox"]
            cname = chair.get("name", cid)

            # Initialise counters on first encounter
            if cid not in self.occupied_counters:
                self.occupied_counters[cid] = 0
                self.empty_counters[cid] = 0
                self.away_timers[cid] = 0.0
                self.prev_status[cid] = "BEKERJA"

            # Check overlap with any tracked person
            occupied = False
            for person in tracked_persons.values():
                iou = compute_iou(cbox, person["bbox"])
                if iou >= iou_thresh:
                    occupied = True
                    break

            # Update frame counters
            if occupied:
                self.occupied_counters[cid] += 1
                self.empty_counters[cid] = 0
            else:
                self.empty_counters[cid] += 1
                self.occupied_counters[cid] = 0

            # Decide status with persistence gating + hysteresis
            if self.occupied_counters[cid] >= persistence:
                new_status = "BEKERJA"
                self.away_timers[cid] = 0.0
            elif self.empty_counters[cid] >= persistence:
                new_status = "TIDAK DI TEMPAT"
                self.away_timers[cid] += dt
            else:
                # In-between: keep previous status (hysteresis)
                new_status = self.prev_status[cid]
                if new_status == "TIDAK DI TEMPAT":
                    self.away_timers[cid] += dt

            # Log status transitions
            if new_status != self.prev_status[cid]:
                if new_status == "TIDAK DI TEMPAT":
                    print(f"[ZONE:{cid}] {cname} -> TIDAK DI TEMPAT (timer mulai)")
                else:
                    dur = format_duration(self.away_timers.get(cid, 0))
                    print(f"[ZONE:{cid}] {cname} -> BEKERJA (away timer was {dur})")

            self.prev_status[cid] = new_status

            # Write results back onto the chair dict for the visualizer to read
            if new_status == "BEKERJA":
                chair["zone_status"] = "BEKERJA"
                chair["zone_color"] = (0, 235, 100)       # Neon Green
                chair["zone_label"] = f"{cname}: BEKERJA"
                chair["is_empty"] = False
                chair["empty_duration"] = 0.0
                chair["empty_label"] = ""
            else:
                dur_str = format_duration(self.away_timers[cid])
                chair["zone_status"] = "TIDAK DI TEMPAT"
                chair["zone_color"] = (0, 50, 255)        # Crimson Red
                chair["zone_label"] = f"{cname}: TIDAK DI TEMPAT {dur_str}"
                chair["is_empty"] = True
                chair["empty_duration"] = self.away_timers[cid]
                chair["empty_label"] = f"TIDAK DI TEMPAT: {dur_str}"
