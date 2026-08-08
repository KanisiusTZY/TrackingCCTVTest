from rules.rule1_on_duty import Rule1OnDuty
from rules.rule2_skiving import Rule2Skiving
from rules.rule3_empty_seat import Rule3EmptySeat
from rules.rule4_opposite_gender import Rule4OppositeGender

class RuleEngine:
    def __init__(self, config):
        self.config = config
        rules_cfg = config.get("rules_enabled", {})

        self.rules = {
            "rule1_on_duty": Rule1OnDuty(enabled=rules_cfg.get("rule1_on_duty", True)),
            "rule2_skiving": Rule2Skiving(enabled=rules_cfg.get("rule2_skiving", True)),
            "rule3_empty_seat": Rule3EmptySeat(enabled=rules_cfg.get("rule3_empty_seat", True)),
            "rule4_opposite_gender": Rule4OppositeGender(enabled=rules_cfg.get("rule4_opposite_gender", True)),
        }

    def process_all(self, tracked_persons, chair_zones, dt):
        """
        Runs all active rules sequentially.
        Rule order matters:
        1. Rule 1 sets default ON_DUTY status.
        2. Rule 2 overrides with SKIVING if reclining persistence met.
        3. Rule 3 checks empty chair zones.
        4. Rule 4 checks opposite gender proximity interactions.
        """
        for rule in self.rules.values():
            rule.process(tracked_persons, chair_zones, self.config, dt)

    def toggle_rule(self, rule_id):
        if rule_id in self.rules:
            new_state = self.rules[rule_id].toggle()
            self.config["rules_enabled"][rule_id] = new_state
            print(f"[RULE ENGINE] {self.rules[rule_id].name} -> {'ENABLED' if new_state else 'DISABLED'}")
            return new_state
        return False

    def is_rule_enabled(self, rule_id):
        if rule_id in self.rules:
            return self.rules[rule_id].enabled
        return False
