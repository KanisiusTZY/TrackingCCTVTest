import cv2
import json
import os
import argparse

class ZoneDrawer:
    def __init__(self, config_path="config.json", video_source="p.mp4", display_width=1280):
        self.config_path = config_path
        self.video_source = video_source
        self.display_width = display_width
        self.chair_zones = []
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.cx, self.cy = -1, -1
        self.scale = 1.0

        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                data = json.load(f)
                self.chair_zones = data.get("chair_zones", [])
        else:
            self.chair_zones = []

    def save_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                data = json.load(f)
        else:
            data = {"thresholds": {}, "rules_enabled": {}}

        data["chair_zones"] = self.chair_zones

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved {len(self.chair_zones)} seat zones to {self.config_path}")

    def mouse_callback(self, event, x, y, flags, param):
        # Map display coordinates back to original frame coordinates
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = orig_x, orig_y
            self.cx, self.cy = orig_x, orig_y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.cx, self.cy = orig_x, orig_y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, orig_x), min(self.iy, orig_y)
            x2, y2 = max(self.ix, orig_x), max(self.iy, orig_y)

            if (x2 - x1) > 15 and (y2 - y1) > 15:
                zone_num = len(self.chair_zones) + 1
                new_zone = {
                    "id": f"chair_{zone_num}",
                    "name": f"Desk Zone {zone_num}",
                    "bbox": [x1, y1, x2, y2]
                }
                self.chair_zones.append(new_zone)
                print(f"[ADDED] {new_zone['name']}: BBox {new_zone['bbox']}")

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {self.video_source}")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("[ERROR] Could not read frame from source.")
            return

        fh, fw = frame.shape[:2]
        if fw > self.display_width:
            self.scale = self.display_width / float(fw)
        else:
            self.scale = 1.0

        window_name = "Skynet Zone Drawer - Click & Drag to add Chair Zone"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(fw * self.scale), int(fh * self.scale))
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n--- INSTRUCTIONS ---")
        print("1. Click and drag left mouse button to draw a seat/chair zone box.")
        print("2. Press 'S' to save zones to config.json.")
        print("3. Press 'C' to clear all drawn zones.")
        print("4. Press 'Q' or ESC to exit.\n")

        while True:
            display_frame = frame.copy()

            # Render existing drawn zones on original frame
            for zone in self.chair_zones:
                x1, y1, x2, y2 = zone["bbox"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    display_frame,
                    f"{zone['id']}: {zone['name']}",
                    (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2
                )

            # Render active drag box
            if self.drawing:
                x1, y1 = min(self.ix, self.cx), min(self.iy, self.cy)
                x2, y2 = max(self.ix, self.cx), max(self.iy, self.cy)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Top instructions banner
            cv2.putText(
                display_frame,
                "Drag to draw chair box | [S] Save | [C] Clear | [Q] Quit",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if self.scale != 1.0:
                render_output = cv2.resize(display_frame, (int(fw * self.scale), int(fh * self.scale)), interpolation=cv2.INTER_AREA)
            else:
                render_output = display_frame

            cv2.imshow(window_name, render_output)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s') or key == ord('S'):
                self.save_config()
            elif key == ord('c') or key == ord('C'):
                self.chair_zones.clear()
                print("[INFO] Cleared all chair zones.")
            elif key == ord('q') or key == ord('Q') or key == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    default_src = "p.mp4" if os.path.exists("p.mp4") else "videoplayback.mp4"
    parser = argparse.ArgumentParser(description="Interactive Seat Zone Drawer")
    parser.add_argument("--source", type=str, default=default_src, help=f"Video file or camera index (default: {default_src})")
    parser.add_argument("--display-width", type=int, default=1280, help="Max display width in pixels (default: 1280)")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    drawer = ZoneDrawer(video_source=src, display_width=args.display_width)
    drawer.run()
