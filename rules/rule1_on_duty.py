from rules.base_rule import BaseRule

class Rule1OnDuty(BaseRule):
    def __init__(self, enabled=True):
        super().__init__(name="Status ON DUTY (在岗)", rule_id="rule1_on_duty", enabled=enabled)

    def process(self, tracked_persons, chair_zones, config, dt):
        if not self.enabled:
            return

        for person_id, person in tracked_persons.items():
            # Apply default ON DUTY status if pose is upright
            if person.get("pose", "upright") == "upright":
                # Only set to ON_DUTY if not currently flagged as SKIVING by Rule 2
                if person.get("recline_counter", 0) < config["thresholds"]["persistence_frames"]:
                    person["status"] = "ON DUTY | 在岗"
                    person["status_color"] = (0, 255, 127)  # Bright Green
