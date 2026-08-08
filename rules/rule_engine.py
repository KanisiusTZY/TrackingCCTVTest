from rules.rule_chair_status import RuleChairStatus

class RuleEngine:
    def __init__(self, config):
        self.config = config
        rules_cfg = config.get("rules_enabled", {})

        self.rules = {
            "rule_chair_status": RuleChairStatus(enabled=rules_cfg.get("rule_chair_status", True)),
        }

    def process_all(self, tracked_objects, dt):
        """
        Runs the dynamic chair status rule on tracked persons and detected chairs.
        tracked_objects: dict {"persons": {...}, "chairs": {...}}
        """
        for rule in self.rules.values():
            rule.process(tracked_objects, self.config, dt)

    def toggle_rule(self, rule_id="rule_chair_status"):
        if rule_id in self.rules:
            new_state = self.rules[rule_id].toggle()
            self.config["rules_enabled"][rule_id] = new_state
            print(f"[RULE ENGINE] {self.rules[rule_id].name} -> {'ENABLED' if new_state else 'DISABLED'}")
            return new_state
        return False

    def is_rule_enabled(self, rule_id="rule_chair_status"):
        if rule_id in self.rules:
            return self.rules[rule_id].enabled
        return False
