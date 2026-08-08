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
        # Track previous status per person_id for console logging on state change
        self._prev_status = {}

    def process(self, tracked_persons, chair_zones, config, dt):
        if not self.enabled:
            return

        persistence_frames = config["thresholds"].get("persistence_frames", 20)

        for person_id, person in tracked_persons.items():
            pose = person.get("pose", "upright")
            method = person.get("pose_method", "unknown")
            old_status = self._prev_status.get(person_id, "upright")

            if pose == "reclining":
                person["recline_counter"] = person.get("recline_counter", 0) + 1

                # Only activate skiving AFTER persistence threshold is met
                if person["recline_counter"] >= persistence_frames:
                    person["idle_timer"] = person.get("idle_timer", 0.0) + dt
                    dur_str = format_duration(person["idle_timer"])
                    person["status"] = f"SKIVING | 旷工: {dur_str}"
                    person["status_color"] = (0, 50, 255)  # Crimson Red

                    # Log status change only on the first frame crossing the threshold
                    if old_status != "reclining":
                        angle_info = person.get("pose_angle", -1)
                        print(f"[ID:{person_id}] status changed: upright -> reclining "
                              f"(method={method}, angle={angle_info}°, "
                              f"after {persistence_frames} consecutive frames)")
                        self._prev_status[person_id] = "reclining"
                # During the persistence buildup, keep current status (don't flip yet)
                # Person retains whatever status they had before (ON DUTY by default)

            else:
                # Pose is "upright": reset counter and timer immediately
                if person.get("recline_counter", 0) > 0 or person.get("idle_timer", 0.0) > 0:
                    if old_status == "reclining":
                        print(f"[ID:{person_id}] status changed: reclining -> upright "
                              f"(method={method}, timer was {format_duration(person.get('idle_timer', 0))})")

                person["recline_counter"] = 0
                person["idle_timer"] = 0.0
                self._prev_status[person_id] = "upright"

        # Clean up stale IDs no longer tracked
        active_ids = set(tracked_persons.keys())
        stale_ids = [pid for pid in self._prev_status if pid not in active_ids]
        for pid in stale_ids:
            del self._prev_status[pid]
