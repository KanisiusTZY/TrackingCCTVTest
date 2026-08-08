import cv2
import math
import numpy as np
import os

class PoseClassifier:
    def __init__(self, hw_ratio_threshold=1.6, angle_upright=60, angle_reclining=40):
        """
        Pose classifier distinguishing:
        - "upright": Duduk tegak/normal (working posture)
        - "reclining": Rebahan/nyender/slouching (not working)

        Primary method: MediaPipe PoseLandmarker keypoints (shoulder-hip torso angle)
        Fallback method: Bounding box aspect ratio (looser threshold 1.6)

        Hysteresis band between angle_reclining and angle_upright prevents
        rapid flip-flopping for borderline poses.
        """
        # Fallback bbox ratio threshold (only used when keypoints unavailable)
        self.hw_ratio_threshold = hw_ratio_threshold

        # Angle thresholds (degrees from horizontal axis)
        # >= angle_upright   -> definitely upright
        # <  angle_reclining -> definitely reclining
        # in between         -> keep previous state (hysteresis)
        self.angle_upright = angle_upright
        self.angle_reclining = angle_reclining

        # Per-person previous pose state for hysteresis (keyed by quantised bbox center)
        self._prev_pose = {}

        # MediaPipe PoseLandmarker initialisation (new Task API for mediapipe >= 1.0)
        self._pose_landmarker = None
        self._mp_available = False
        self._mp_module = None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python import BaseOptions

            self._mp_module = mp

            # Find model file — check working dir and common locations
            model_path = None
            candidates = [
                "pose_landmarker_lite.task",
                os.path.join(os.path.dirname(__file__), "..", "pose_landmarker_lite.task"),
                os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    model_path = os.path.abspath(c)
                    break

            if model_path is None:
                # Auto-download the model
                print("[POSE] pose_landmarker_lite.task not found. Downloading...")
                import urllib.request
                url = ("https://storage.googleapis.com/mediapipe-models/"
                       "pose_landmarker/pose_landmarker_lite/float16/latest/"
                       "pose_landmarker_lite.task")
                model_path = "pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, model_path)
                print(f"[POSE] Downloaded model to {os.path.abspath(model_path)}")

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_poses=1,
            )
            self._pose_landmarker = PoseLandmarker.create_from_options(options)
            self._mp_available = True
            print(f"[POSE] MediaPipe PoseLandmarker initialised (model: {os.path.basename(model_path)}).")

        except Exception as e:
            print(f"[POSE WARNING] MediaPipe PoseLandmarker unavailable ({e}). Using bbox fallback only.")

    def _compute_torso_angle(self, frame, person_bbox):
        """
        Use MediaPipe PoseLandmarker to get shoulder & hip keypoints,
        then compute the torso angle relative to horizontal.

        Returns (angle_degrees, success_bool).
        angle_degrees: angle of the shoulder-midpoint → hip-midpoint line
                       measured from the HORIZONTAL axis.
                       ~90° = perfectly upright, ~0° = lying flat.
        """
        if not self._mp_available or self._pose_landmarker is None:
            return 0.0, False

        x1, y1, x2, y2 = person_bbox
        h_frame, w_frame = frame.shape[:2]

        # Clamp bbox to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 25:
            return 0.0, False

        try:
            mp = self._mp_module
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            result = self._pose_landmarker.detect(mp_image)
        except Exception:
            return 0.0, False

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return 0.0, False

        lm = result.pose_landmarks[0]
        crop_h, crop_w = crop.shape[:2]

        # MediaPipe landmark indices:
        # 11 = left shoulder, 12 = right shoulder
        # 23 = left hip,      24 = right hip
        ls = lm[11]  # left shoulder
        rs = lm[12]  # right shoulder
        lh = lm[23]  # left hip
        rh = lm[24]  # right hip

        # Check visibility — all four must be reasonably visible
        min_vis = 0.3
        if (ls.visibility < min_vis or rs.visibility < min_vis or
                lh.visibility < min_vis or rh.visibility < min_vis):
            return 0.0, False

        # Midpoint of shoulders (normalised coords [0..1])
        mid_shoulder_x = (ls.x + rs.x) / 2.0 * crop_w
        mid_shoulder_y = (ls.y + rs.y) / 2.0 * crop_h

        # Midpoint of hips
        mid_hip_x = (lh.x + rh.x) / 2.0 * crop_w
        mid_hip_y = (lh.y + rh.y) / 2.0 * crop_h

        # Vector from hip-midpoint to shoulder-midpoint
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_hip_y - mid_shoulder_y  # Inverted Y (image coords: y increases downward)

        # Angle from horizontal (atan2 gives angle from positive X axis)
        angle_rad = math.atan2(dy, abs(dx) if abs(dx) > 0.001 else 0.001)
        angle_deg = math.degrees(angle_rad)

        # Clamp to [0, 90] range
        angle_deg = max(0.0, min(90.0, angle_deg))

        return angle_deg, True

    def classify(self, frame, person_bbox, chair_zone_bbox=None):
        """
        Evaluates person posture inside frame.
        Returns dict: {"pose": "upright" / "reclining", "score": float, "method": str}
        """
        x1, y1, x2, y2 = person_bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Generate a stable key for hysteresis lookup based on bbox centre
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Quantise to 50px grid so slight bbox jitter maps to same person
        hysteresis_key = (cx // 50, cy // 50)
        prev_pose = self._prev_pose.get(hysteresis_key, "upright")

        # --- Primary method: MediaPipe PoseLandmarker keypoint angle ---
        angle, kp_success = self._compute_torso_angle(frame, person_bbox)

        if kp_success:
            if angle >= self.angle_upright:
                pose_str = "upright"
            elif angle < self.angle_reclining:
                pose_str = "reclining"
            else:
                # Hysteresis band: keep previous state to avoid flicker
                pose_str = prev_pose

            self._prev_pose[hysteresis_key] = pose_str
            return {
                "pose": pose_str,
                "score": round(angle / 90.0, 2),
                "angle": round(angle, 1),
                "method": "mediapipe",
                "aspect_ratio": round(h / float(w), 2),
            }

        # --- Fallback method: bbox aspect ratio (looser threshold) ---
        aspect_ratio = h / float(w)

        # Use the looser fallback threshold (default 1.6 instead of old 1.25)
        is_reclining = aspect_ratio < self.hw_ratio_threshold

        pose_str = "reclining" if is_reclining else "upright"
        self._prev_pose[hysteresis_key] = pose_str

        confidence_score = abs(aspect_ratio - self.hw_ratio_threshold) / self.hw_ratio_threshold

        return {
            "pose": pose_str,
            "score": round(confidence_score, 2),
            "angle": -1.0,  # Indicates keypoints were not available
            "method": "bbox_fallback",
            "aspect_ratio": round(aspect_ratio, 2),
        }

    def release(self):
        """Release MediaPipe resources."""
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
