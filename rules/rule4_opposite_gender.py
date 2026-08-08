import math
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

class Rule4OppositeGender(BaseRule):
    def __init__(self, enabled=True):
        super().__init__(name="Opposite Gender Interaction (异性交谈时长)", rule_id="rule4_opposite_gender", enabled=enabled)
        self.pair_timers = {}  # tuple(idA, idB) -> seconds

    def process(self, tracked_persons, chair_zones, config, dt):
        if not self.enabled:
            return

        prox_thresh = config["thresholds"]["proximity_interaction_px"]
        person_ids = list(tracked_persons.keys())
        active_pairs = set()

        # Reset interaction flags on all persons initially
        for person in tracked_persons.values():
            person["interaction_active"] = False
            person["interaction_timer"] = 0.0
            person["interaction_partner_id"] = None
            person["interaction_partner_gender"] = None

        # Iterate over all pairs
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                idA = person_ids[i]
                idB = person_ids[j]

                pA = tracked_persons[idA]
                pB = tracked_persons[idB]

                # Condition A: Gender must be different
                genderA = pA.get("gender", "Unknown")
                genderB = pB.get("gender", "Unknown")

                if genderA != genderB and genderA != "Unknown" and genderB != "Unknown":
                    # Condition B: Distance below threshold
                    cxA, cyA = pA["centroid"]
                    cxB, cyB = pB["centroid"]
                    dist = math.hypot(cxA - cxB, cyA - cyB)

                    pair_key = (min(idA, idB), max(idA, idB))

                    if dist < prox_thresh:
                        active_pairs.add(pair_key)
                        if pair_key not in self.pair_timers:
                            self.pair_timers[pair_key] = 0.0
                        
                        self.pair_timers[pair_key] += dt
                        dur_sec = self.pair_timers[pair_key]
                        dur_str = format_duration(dur_sec)

                        # Set interaction state on person A
                        pA["interaction_active"] = True
                        pA["interaction_timer"] = dur_sec
                        pA["interaction_label"] = f"OPPOSITE GENDER CHAT | 异性交谈: {dur_str}"
                        pA["interaction_partner_id"] = idB
                        pA["interaction_partner_gender"] = genderB

                        # Set interaction state on person B
                        pB["interaction_active"] = True
                        pB["interaction_timer"] = dur_sec
                        pB["interaction_label"] = f"OPPOSITE GENDER CHAT | 异性交谈: {dur_str}"
                        pB["interaction_partner_id"] = idA
                        pB["interaction_partner_gender"] = genderA
                    else:
                        # Reset timer if distance exceeds threshold
                        if pair_key in self.pair_timers:
                            self.pair_timers[pair_key] = 0.0

        # Clean up stale pairs no longer tracked
        stale_keys = [k for k in self.pair_timers.keys() if k not in active_pairs]
        for k in stale_keys:
            self.pair_timers[k] = 0.0
