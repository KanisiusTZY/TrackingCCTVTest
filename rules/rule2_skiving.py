from rules.base_rule import BaseRule

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

class Rule2Skiving(BaseRule):
    def __init__(self, enabled=True):
        super().__init__(name="Idle/Skiving Timer (旷工时长)", rule_id="rule2_skiving", enabled=enabled)

    def process(self, tracked_persons, chair_zones, config, dt):
        if not self.enabled:
            return

        persistence_frames = config["thresholds"]["persistence_frames"]

        for person_id, person in tracked_persons.items():
            pose = person.get("pose", "upright")

            if pose == "reclining":
                person["recline_counter"] = person.get("recline_counter", 0) + 1
                
                # Check persistence threshold before activating counter
                if person["recline_counter"] >= persistence_frames:
                    person["idle_timer"] = person.get("idle_timer", 0.0) + dt
                    dur_str = format_duration(person["idle_timer"])
                    person["status"] = f"SKIVING | 旷工: {dur_str}"
                    person["status_color"] = (0, 50, 255)  # Crimson Red
            else:
                # Reset counter and timer only when pose strictly returns to upright
                person["recline_counter"] = 0
                person["idle_timer"] = 0.0
