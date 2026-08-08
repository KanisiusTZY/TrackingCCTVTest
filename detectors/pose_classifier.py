import cv2
import numpy as np

class PoseClassifier:
    def __init__(self, hw_ratio_threshold=1.25):
        """
        Pose classifier distinguishing:
        - "upright": Duduk tegak/normal
        - "reclining": Rebahan/nyender/slouching
        """
        self.hw_ratio_threshold = hw_ratio_threshold

    def classify(self, frame, person_bbox, chair_zone_bbox=None):
        """
        Evaluates person posture inside frame & optional chair zone.
        Returns dict: {"pose": "upright" / "reclining", "score": float}
        """
        x1, y1, x2, y2 = person_bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        aspect_ratio = h / float(w)

        # Head position analysis: upper third vs lower region
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size > 0:
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            # Find vertical mass distribution (centroid of upper body intensity)
            vertical_projection = np.sum(255 - gray, axis=1)
            total_mass = np.sum(vertical_projection)
            if total_mass > 0:
                y_center_mass = np.sum(np.arange(len(vertical_projection)) * vertical_projection) / float(total_mass)
                relative_y_center = y_center_mass / float(h)
            else:
                relative_y_center = 0.5
        else:
            relative_y_center = 0.5

        # Reclining condition check:
        # 1. Low height/width aspect ratio (person leaning horizontally)
        # 2. Or mass center shifted downward (slouching low in chair)
        is_reclining = (aspect_ratio < self.hw_ratio_threshold) or (relative_y_center > 0.62)

        pose_str = "reclining" if is_reclining else "upright"
        confidence_score = abs(aspect_ratio - self.hw_ratio_threshold) / self.hw_ratio_threshold

        return {
            "pose": pose_str,
            "score": round(confidence_score, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "relative_y_center": round(relative_y_center, 2)
        }
