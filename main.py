import cv2
import json
import time
import os
import argparse

from tracker import CentroidTracker
from detectors.person_detector import PersonDetector
from detectors.face_gender_detector import FaceGenderDetector
from detectors.pose_classifier import PoseClassifier
from rules.rule_engine import RuleEngine
from visualizer import Visualizer

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        print(f"[WARNING] Config file {config_path} not found. Creating default configuration.")
        default_config = {
            "chair_zones": [
                {"id": "chair_1", "name": "Desk Zone 1", "bbox": [100, 200, 350, 500]},
                {"id": "chair_2", "name": "Desk Zone 2", "bbox": [450, 200, 700, 500]}
            ],
            "thresholds": {
                "iou_chair_occupied": 0.20,
                "proximity_interaction_px": 180,
                "persistence_frames": 15,
                "recline_hw_ratio": 1.25
            },
            "rules_enabled": {
                "rule1_on_duty": True,
                "rule2_skiving": True,
                "rule3_empty_seat": True,
                "rule4_opposite_gender": True
            }
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    with open(config_path, "r") as f:
        return json.load(f)

def main():
    default_src = "p.mp4" if os.path.exists("p.mp4") else "videoplayback.mp4"

    parser = argparse.ArgumentParser(description="Skynet Office CCTV Computer Vision Monitoring System")
    parser.add_argument("--source", type=str, default=default_src, help=f"Video source file or camera index (default: {default_src})")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file (default: config.json)")
    parser.add_argument("--output", type=str, default="output_skynet_monitoring.mp4", help="Path to save output video (default: output_skynet_monitoring.mp4)")
    parser.add_argument("--display-width", type=int, default=1280, help="Display window max width in pixels (default: 1280)")
    parser.add_argument("--headless", action="store_true", help="Run without rendering GUI display window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after processing max frames (0 = unlimited)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    chair_zones = config.get("chair_zones", [])

    # Initialize detection & tracking components
    print("[INFO] Initializing Skynet Computer Vision modules...")
    person_detector = PersonDetector(confidence_threshold=0.35)
    face_gender_detector = FaceGenderDetector()
    pose_classifier = PoseClassifier(hw_ratio_threshold=config["thresholds"].get("recline_hw_ratio", 1.25))
    tracker = CentroidTracker(max_disappeared=30)
    rule_engine = RuleEngine(config)
    visualizer = Visualizer()

    # Open video source
    video_source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {video_source}")
        return

    # Setup Video Writer if output path provided
    video_writer = None
    if args.output:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 120:
            video_fps = 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_w == 0 or frame_h == 0:
            frame_w, frame_h = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, video_fps, (frame_w, frame_h))
        print(f"[INFO] Recording output video to '{args.output}' ({frame_w}x{frame_h} @ {video_fps:.1f} FPS)...")

    window_name = "Skynet Office Monitoring System"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"[INFO] Skynet Monitoring Engine running on source '{args.source}'...")
    print("Press [1], [2], [3], [4] to toggle rules | [R] Reset Timers | [Q] Quit\n")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video if source is a file and not recording, or stop if finished recording
            if isinstance(video_source, str) and os.path.exists(video_source) and args.max_frames == 0 and not args.output:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        fps = 1.0 / dt if dt > 0 else 0.0
        frame_count += 1

        # Step 1: Detect persons
        person_bboxes = person_detector.detect(frame)

        # Step 2: Detect faces/genders and classify poses per person box
        face_infos = []
        pose_infos = []

        for p_bbox in person_bboxes:
            f_info = face_gender_detector.detect_and_classify(frame, p_bbox)
            p_info = pose_classifier.classify(frame, p_bbox)

            face_infos.append(f_info)
            pose_infos.append(p_info)

        # Step 3: Update multi-object tracker
        tracked_persons = tracker.update(person_bboxes, face_infos, pose_infos)

        # Step 4: Process all active rules
        rule_engine.process_all(tracked_persons, chair_zones, dt)

        # Step 5: Render Visual HUD Overlay
        output_frame = visualizer.render(frame, tracked_persons, chair_zones, rule_engine, fps=fps)

        # Write frame to MP4 output video file
        if video_writer is not None:
            video_writer.write(output_frame)

        if not args.headless:
            # Scale frame for comfortable display without screen overflow/zoom
            fh, fw = output_frame.shape[:2]
            target_w = args.display_width
            if fw > target_w:
                scale = target_w / float(fw)
                disp_w = target_w
                disp_h = int(fh * scale)
                display_render = cv2.resize(output_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                display_render = output_frame

            cv2.imshow(window_name, display_render)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):
                rule_engine.toggle_rule("rule1_on_duty")
            elif key == ord('2'):
                rule_engine.toggle_rule("rule2_skiving")
            elif key == ord('3'):
                rule_engine.toggle_rule("rule3_empty_seat")
            elif key == ord('4'):
                rule_engine.toggle_rule("rule4_opposite_gender")
            elif key == ord('r') or key == ord('R'):
                print("[RESET] Resetting all rule timers...")
                for p in tracked_persons.values():
                    p["idle_timer"] = 0.0
                    p["recline_counter"] = 0
                for rule in rule_engine.rules.values():
                    if hasattr(rule, "empty_timers"):
                        rule.empty_timers.clear()
                        rule.empty_counters.clear()
                    if hasattr(rule, "pair_timers"):
                        rule.pair_timers.clear()
            elif key == ord('s') or key == ord('S'):
                snap_path = f"snapshot_frame_{frame_count}.jpg"
                cv2.imwrite(snap_path, output_frame)
                print(f"[SNAPSHOT] Saved frame to {snap_path}")
            elif key == ord('q') or key == ord('Q') or key == 27:
                print("[INFO] User terminated session.")
                break

        if args.max_frames > 0 and frame_count >= args.max_frames:
            print(f"[INFO] Reached max frames limit ({args.max_frames}). Stopping.")
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"[RECORDING SAVED] Output video saved to: {os.path.abspath(args.output)}")

    cv2.destroyAllWindows()
    if 'output_frame' in locals() and output_frame is not None:
        cv2.imwrite("skynet_monitoring_preview.jpg", output_frame)
        print(f"[PREVIEW] Saved demo preview snapshot to d:\\Monitoring\\skynet_monitoring_preview.jpg")
    print("[INFO] Skynet Monitoring Engine stopped.")

if __name__ == "__main__":
    main()
