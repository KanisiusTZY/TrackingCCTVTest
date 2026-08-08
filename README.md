# 🛡️ Skynet CCTV Dynamic Chair & Upper-Body Monitoring System

A modular, real-time Computer Vision office CCTV monitoring system built in Python using OpenCV, YOLOv8 (`ultralytics`), dynamic chair object detection, upper-body person cropping, and multi-category tracking.

Inspired by the "Skynet" office monitoring system, it features dynamic seat detection (no manual zone drawing required) and real-time seat status monitoring.

---

## ⚡ Key Features

- **Dynamic Chair Object Detection**: Automatically detects chairs in real-time using YOLOv8 (COCO class 56).
- **Upper-Body Person Crop**: Crops person detections to upper-body (`person_upper_body_ratio = 0.55`) for accurate IoU matching against chairs and clean visual bounding boxes.
- **Multi-Category Tracking**: Tracks persons and chairs with independent IDs. Includes a 10-frame last-known bbox persistence mechanism for chairs to prevent flicker during temporary occlusion.
- **Seat Status Logic**:
  - **🟢 BEKERJA**: Tight green bounding box around the upper body of an occupied chair.
  - **🔴 TIDAK DI TEMPAT**: Tight red bounding box around an empty chair with an active real-time duration counter (`TIDAK DI TEMPAT: XmYYs`).

---

## 📁 Repository Structure

```
d:\Monitoring
├── config.json                     # Dynamic thresholds (IoU, persistence, upper-body ratio)
├── main.py                         # Main CCTV monitoring application entrypoint
├── visualizer.py                   # OpenCV Skynet Cyberpunk HUD & badge renderer
├── tracker.py                      # MultiCategoryTracker (independent Person & Chair tracking)
├── detectors/
│   └── person_detector.py          # YOLOv8 Person & Chair detector with upper-body crop
└── rules/
    ├── base_rule.py                # Abstract Base Class for modular rules
    ├── rule_chair_status.py        # Dynamic chair occupancy & away timer rule
    └── rule_engine.py              # Master rule manager & orchestrator
```

---

## ⚙️ Configuration (`config.json`)

```json
{
  "thresholds": {
    "iou_chair_occupied": 0.15,
    "persistence_frames": 15,
    "person_upper_body_ratio": 0.55
  }
}
```

---

## 🎮 Usage Guide

### 1. Run Real-Time Skynet Monitoring
```bash
python main.py --source p.mp4
```
*To use a live webcam, pass `--source 0`.*

### 2. Keyboard Hotkeys
- **`R`**: Reset all active away timers
- **`S`**: Save snapshot frame to disk
- **`Q`**: Exit application

---

## 🎬 Output Video Recording
The system automatically records the processed video stream with all active overlays and timers saved directly to `output_skynet_monitoring.mp4`.
