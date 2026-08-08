import cv2
import numpy as np

# Color Palette (BGR)
COLOR_BEKERJA = (0, 235, 100)       # Bright Neon Green
COLOR_AWAY = (0, 50, 255)          # Crimson Red
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_HUD_BG = (15, 18, 25)

def draw_corner_box(img, bbox, color, thickness=2, corner_len=15):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

    # Accent corners
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_filled_badge(img, text, position, bg_color, text_color=COLOR_TEXT_WHITE, font_scale=0.55, pad=6):
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, 2)

    rect_x1 = x
    rect_y1 = max(0, y - text_h - (pad * 2))
    rect_x2 = x + text_w + (pad * 2)
    rect_y2 = y

    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, 2)

    cv2.putText(
        img,
        text,
        (x + pad, y - pad),
        font,
        font_scale,
        text_color,
        2,
        cv2.LINE_AA
    )
    return rect_y1


class Visualizer:
    def __init__(self):
        pass

    def render(self, frame, registered_chairs, rule_engine, fps=0.0):
        out = frame.copy()
        h, w = out.shape[:2]

        bekerja_count = 0
        away_count = 0

        # Render each chair entry in ChairRegistry
        for chair_id, chair in registered_chairs.items():
            status = chair.get("status", "TIDAK DI TEMPAT")

            if status == "BEKERJA":
                bekerja_count += 1
                upper_bbox = chair.get("matched_upper_body_bbox")
                if upper_bbox is not None:
                    px1, py1, px2, py2 = upper_bbox
                    # Tight green box around person upper body
                    draw_corner_box(out, (px1, py1, px2, py2), COLOR_BEKERJA, thickness=2)
                    draw_filled_badge(out, "BEKERJA", (px1, py1), COLOR_BEKERJA, font_scale=0.55)
            else:
                away_count += 1
                cx1, cy1, cx2, cy2 = chair["bbox"]
                away_label = chair.get("away_label", "TIDAK DI TEMPAT")

                # Tight red box around registered chair position
                overlay = out.copy()
                cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), COLOR_AWAY, -1)
                cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)
                cv2.rectangle(out, (cx1, cy1), (cx2, cy2), COLOR_AWAY, 2)

                draw_filled_badge(out, away_label, (cx1, cy1), COLOR_AWAY, font_scale=0.55)

        # Top Skynet HUD Bar
        hud_h = 45
        hud_overlay = out.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(hud_overlay, 0.88, out, 0.12, 0, out)
        cv2.line(out, (0, hud_h), (w, hud_h), (0, 235, 100), 1)

        sys_title = "SKYNET CHAIR REGISTRY SYSTEM"
        fps_text = f"FPS: {fps:.1f}"
        status_stats = f"BEKERJA: {bekerja_count}  |  TIDAK DI TEMPAT: {away_count}"

        cv2.putText(out, sys_title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 235, 100), 2)
        cv2.putText(out, fps_text, (380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(out, status_stats, (480, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2)

        # Bottom Controls Legend Bar
        legend_str = "STATUS: [GREEN] BEKERJA (Upper Body) | [RED] TIDAK DI TEMPAT (Chair Registry) | [R] Reset Timers | [Q] Quit"
        cv2.putText(out, legend_str, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return out
