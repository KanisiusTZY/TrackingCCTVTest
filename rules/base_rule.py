from abc import ABC, abstractmethod

class BaseRule(ABC):
    def __init__(self, name, rule_id, enabled=True):
        self.name = name
        self.rule_id = rule_id
        self.enabled = enabled

    @abstractmethod
    def process(self, tracked_persons, chair_zones, config, dt):
        """
        Process the current frame state for tracked persons and chair zones.
        tracked_persons: dict of {person_id: person_data}
        chair_zones: list of chair zone dicts
        config: global configuration dict
        dt: delta time in seconds elapsed since last frame
        """
        pass

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
