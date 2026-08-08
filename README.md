# 🛡️ Skynet Office CCTV Computer Vision Monitoring System

A modular, real-time Computer Vision office CCTV monitoring system built in Python using OpenCV, YOLOv8 (`ultralytics`), multi-object tracking, face/gender detection, and posture classification.

Inspired by the "Skynet" office monitoring system, it features a Cyberpunk HUD overlay and implements 4 core workplace monitoring rules with real-time timers.

---

## ⚡ Key Features & 4 Core Monitoring Rules

### 🟢 RULE 1 - STATUS "ON DUTY" (在岗) [Green Box]
- **Condition**: Face detected AND person in upright sitting posture facing desk inside chair zone.
- **State**: Default state (no active timer).
- **Overlay**: Bright green bounding box + top status badge `[ON DUTY | 在岗]`.

### 🔴 RULE 2 - IDLE / SKIVING TIMER (旷工时长) [Red Box]
- **Condition**: Person inside seat zone with detected posture `"reclining"` (slouching/leaning back) for $N \ge 15$ consecutive frames.
- **Logic**: Increments `idle_timer += dt`. Micro-movements do NOT reset timer. Timer resets to 0 only when posture strictly returns to `"upright"`.
- **Overlay**: Crimson red box + real-time duration badge `[SKIVING | 旷工: XhYYmin]`.

### 🔴 RULE 3 - EMPTY SEAT TIMER (离开座位时长) [Red Box on Chair]
- **Condition**: Chair zone detected empty (no person bounding box overlap, $\text{IoU} < 0.20$) for $N \ge 15$ consecutive frames.
- **Logic**: Increments `empty_timer[chair] += dt`. Resets to 0 immediately when person occupies seat.
- **Overlay**: Red outline + glowing overlay + `[MENINGGALKAN KURSI: XmYYs]`.

### 🔴 RULE 4 - OPPOSITE GENDER INTERACTION TIMER (异性交谈时长) [Red Box above 2 Persons]
- **Condition**: Two persons $A, B$ with different genders (`gender_A != gender_B`) AND proximity distance $< 180\text{px}$.
- **Logic**: Increments `interaction_timer[pair] += dt`. Resets/pauses immediately when pair separates or distance exceeds threshold.
- **Overlay**: Dual red badges above both heads with synchronized timer `[OPPOSITE GENDER CHAT: XmYYs]`.

---

## 📁 Repository Structure

```
d:\Monitoring
├── config.json                     # Seat zones, thresholds, and rule enable flags
├── main.py                         # Main CCTV monitoring application entrypoint
├── zone_drawer.py                  # Interactive mouse ROI drawing tool for seat zones
├── visualizer.py                   # OpenCV Skynet Cyberpunk HUD & badge renderer
├── tracker.py                      # Multi-Object Centroid Tracker (persistent Person IDs)
├── detectors/
│   ├── person_detector.py          # YOLOv8 Person detector with OpenCV HOG fallback
│   ├── face_gender_detector.py     # Face detection & gender classifier
│   └── pose_classifier.py          # Upright vs Reclining posture classifier
└── rules/
    ├── base_rule.py                # Abstract Base Class for modular rules
    ├── rule1_on_duty.py            # Rule 1 implementation
    ├── rule2_skiving.py            # Rule 2 implementation
    ├── rule3_empty_seat.py         # Rule 3 implementation
    ├── rule4_opposite_gender.py    # Rule 4 implementation
    └── rule_engine.py              # Master rule manager & orchestrator
```

---

## 🛠️ Requirements & Installation

1. **Clone the repository**:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd Monitoring
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python ultralytics numpy torch scipy
   ```

---

## 🎮 Usage Guide

### 1. Interactively Draw Seat/Chair Zones (Optional)
Run the zone drawer tool to define desk ROI coordinates using mouse clicks:
```bash
python zone_drawer.py --source p.mp4
```
- **Click & Drag** left mouse button to draw a seat box.
- Press **`S`** to save zones to `config.json`.
- Press **`Q`** to exit.

### 2. Run Real-Time Skynet Monitoring
Run the monitoring engine on a video file or live webcam stream:
```bash
python main.py --source p.mp4
```
*To use a live webcam, pass `--source 0`.*

### 3. Keyboard Hotkeys
- **`1`**: Toggle **Rule 1 (Status On Duty)** ON / OFF
- **`2`**: Toggle **Rule 2 (Skiving Timer)** ON / OFF
- **`3`**: Toggle **Rule 3 (Empty Seat Timer)** ON / OFF
- **`4`**: Toggle **Rule 4 (Opposite Gender Interaction Timer)** ON / OFF
- **`R`**: Reset all active rule timers
- **`S`**: Save snapshot frame to disk
- **`Q`**: Exit application

---

## 🎬 Output Video Recording
The system automatically records the processed video stream with all active overlays and timers saved directly to `output_skynet_monitoring.mp4`.
