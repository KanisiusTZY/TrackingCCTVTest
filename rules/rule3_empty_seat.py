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

class Rule3EmptySeat(BaseRule):
    def __init__(self, enabled=True):
        super().__init__(name="Empty Seat Timer (离开座位时长)", rule_id="rule3_empty_seat", enabled=enabled)
        self.empty_counters = {}  # chair_id -> frame_count
        self.empty_timers = {}    # chair_id -> total_seconds

    def process(self, tracked_persons, chair_zones, config, dt):
        if not self.enabled:
            return

        iou_thresh = config["thresholds"]["iou_chair_occupied"]
        persistence_frames = config["thresholds"]["persistence_frames"]

        for chair in chair_zones:
            chair_id = chair["id"]
            chair_bbox = chair["bbox"]

            if chair_id not in self.empty_counters:
                self.empty_counters[chair_id] = 0
                self.empty_timers[chair_id] = 0.0

            # Check if any person overlaps with chair zone
            occupied = False
            for person in tracked_persons.values():
                p_bbox = person["bbox"]
                iou = compute_iou(chair_bbox, p_bbox)
                if iou >= iou_thresh:
                    occupied = True
                    break

            if not occupied:
                self.empty_counters[chair_id] += 1
                if self.empty_counters[chair_id] >= persistence_frames:
                    self.empty_timers[chair_id] += dt
                    chair["is_empty"] = True
                    chair["empty_duration"] = self.empty_timers[chair_id]
                    chair["empty_label"] = f"MENINGGALKAN KURSI: {format_duration(self.empty_timers[chair_id])}"
                else:
                    chair["is_empty"] = False
                    chair["empty_duration"] = 0.0
                    chair["empty_label"] = ""
            else:
                # Reset timer immediately when occupied
                self.empty_counters[chair_id] = 0
                self.empty_timers[chair_id] = 0.0
                chair["is_empty"] = False
                chair["empty_duration"] = 0.0
                chair["empty_label"] = ""
