from rules.rule_seat_status import RuleSeatStatus

class RuleEngine:
    def __init__(self, config):
        self.config = config
        rules_cfg = config.get("rules_enabled", {})

        self.rules = {
            "rule_seat_status": RuleSeatStatus(enabled=rules_cfg.get("rule_seat_status", True)),
        }

    def process_all(self, tracked_persons, chair_zones, dt):
        """
        Runs the active seat status rule.
        """
        for rule in self.rules.values():
            rule.process(tracked_persons, chair_zones, self.config, dt)

    def toggle_rule(self, rule_id="rule_seat_status"):
        if rule_id in self.rules:
            new_state = self.rules[rule_id].toggle()
            self.config["rules_enabled"][rule_id] = new_state
            print(f"[RULE ENGINE] {self.rules[rule_id].name} -> {'ENABLED' if new_state else 'DISABLED'}")
            return new_state
        return False

    def is_rule_enabled(self, rule_id="rule_seat_status"):
        if rule_id in self.rules:
            return self.rules[rule_id].enabled
        return False
